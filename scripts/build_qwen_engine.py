"""Build a fully triton-cpu-driven, chained, real-weight, autoregressive Qwen2.5-0.5B
inference engine for riscv64 (BPI-F3): every operator is dispatched via a triton-cpu
riscv64-compiled kernel .so, chained together (real buffer aliasing, not isolated
per-kernel tests), with a real StaticCache so the decode-step kernel/grid sequence is
IDENTICAL across positions and can be compiled once and replayed for every token.

Phase A (this process, native x86 target): run one prefill forward + two decode-step
forwards through torch.compile(backend="inductor", cpu_backend="triton"), hooking
CPULauncher to capture every kernel launch's real grid + per-argument (storage base
addr, byte offset, byte size, dtype, role) for both passes.

Classification (see OPEN_QUESTIONS.md "on-device chained engine" section for the
reasoning this is based on):
  - written  = any storage ever targeted by an out_ptr/in_out_ptr arg.
  - weight   = never written, AND same VALUE in decode0 vs decode1 -> true constant,
               dump real bytes once.
  - step_input = never written, but VALUE differs between decode0 and decode1 -> a
               per-step external input (current cache position / current token id);
               matched by value against the known L, L+1, tok0, tok1 host-side facts.
  - kv_state = written, AND same base address in prefill capture and decode0 capture
               -> persists across the whole generation (StaticCache k/v buffers);
               needs one real persistent C buffer, written by prefill once, then
               read+written every decode step.
  - scratch  = everything else (written, but local to one pass) -> one persistent C
               buffer per unique address observed *within* that one pass's own
               capture (prefill scratch and decode-template scratch are separate
               pools), reused/overwritten every replay -- matches Inductor's own
               buffer-planning semantics exactly since the real computation already
               fully overwrites what it needs each call.

Phase B: for every unique kernel (by id(src.fn)), reconstruct + compile to riscv64 via
riscv.py's compile_kernel_to_so, and dump metadata (real symbol name + real extern C
signature) for the driver generator.

Phase C (scripts/gen_qwen_driver.py, separate process): consumes the manifest this
script writes and emits+builds driver.c.
"""
import os
os.environ["TRITON_DEFAULT_BACKEND"] = "cpu"
os.environ["TRITON_CPU_TARGET"] = "native"
import triton
import triton.language as tl
triton.runtime.driver.set_active_to_cpu()
import torch
import torch._dynamo.device_interface as _di

# Upstream gap: CpuInterface never overrides DeviceInterface.device (the inner
# context-manager class used to switch CUDA device index), so it inherits the
# base's NotImplementedError placeholder. max_autotune_gemm's template-
# benchmarking path calls it unconditionally regardless of device type,
# crashing on CPU. Without max_autotune_gemm, Inductor's CPU backend routes
# matmuls (the actual q/k/v/o_proj, gate/up/down_proj, lm_head GEMMs -- i.e.
# almost all the FLOPs) through plain ATen/BLAS, never through triton-cpu at
# all, silently violating "every operation must flow through triton-cpu". A
# no-op context manager is a correct substitute for CPU (there is no device
# index to switch).
class _NoOpDevice:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_di.CpuInterface.device = _NoOpDevice

import torch._inductor.config as inductor_config
inductor_config.cpu_backend = "triton"
# Autotuning stays OFF (max_autotune / max_autotune_gemm default to False): timing
# candidate GEMM tile configs on this x86 host says nothing about riscv64. GEMMs
# (mm/addmm/bmm) still lower to Triton templates because the installed torch/_inductor
# is patched by scripts/patch_inductor_cpu_triton_gemm.py (run it once after installing
# torch). The tile config is chosen by rule: env TRITON_CPU_GEMM_CFG.
inductor_config.max_autotune_gemm = False
inductor_config.max_autotune = False

import triton.backends.cpu.driver as cpu_driver

ENGINE_DIR = os.path.join(os.environ["HOME"], "qwen_triton_engine")
os.makedirs(ENGINE_DIR, exist_ok=True)

MAX_CACHE_LEN = int(os.environ.get("QWEN_MAX_CACHE_LEN", "128"))
PROMPT = os.environ.get(
    "QWEN_PROMPT", "Please briefly explain what Time to First Token (TTFT) is.")
NUM_DECODE_SAMPLES = 2  # decode0, decode1 -- enough to classify persistent vs transient

# ---------------------------------------------------------------------------
# Phase A: capture
# ---------------------------------------------------------------------------
captured_srcs = {}  # kernel_id -> ASTSource (first occurrence, for later compile)
current_calls = None  # set to a list while capturing one pass
first_snapshot = {}  # addr -> full torch.Tensor clone, captured the first time we see this address


def capture_arg_meta(a):
    if isinstance(a, torch.Tensor):
        t = a.detach()
        storage = t.untyped_storage()
        addr = storage.data_ptr()
        if addr not in first_snapshot:
            # Snapshot the ENTIRE underlying storage (offset 0 to storage.nbytes()),
            # not just this one view -- different args can reference the same base
            # storage at different offsets, and we key/allocate by base address, so
            # one canonical whole-storage snapshot lets every access apply its own
            # offset against it (byte-for-byte matching what this arg's own offset
            # arithmetic could legitimately reach).
            elem_size = t.element_size()
            total_elems = storage.nbytes() // elem_size
            flat = torch.empty(0, dtype=t.dtype)
            flat.set_(storage, 0, (total_elems,), (1,))
            first_snapshot[addr] = flat.clone()
        return ("ptr", addr, t.storage_offset() * t.element_size(),
                storage.nbytes(), str(t.dtype), t.flatten().tolist() if t.numel() <= 8 else None)
    return ("scalar", a)


_orig_init = cpu_driver.CPULauncher.__init__
_orig_call = cpu_driver.CPULauncher.__call__


def capturing_init(self, src, metadata):
    _orig_init(self, src, metadata)
    self._capture_name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    self._capture_kernel_id = id(src.fn)
    self._capture_arg_names = [n for n in src.fn.arg_names if src.signature.get(n) != "constexpr"]
    self._capture_constexprs = dict(src.constants) if hasattr(src, "constants") else {}
    if self._capture_kernel_id not in captured_srcs:
        captured_srcs[self._capture_kernel_id] = src


def capturing_call(self, *args, **kwargs):
    if current_calls is not None:
        kernel_args = args[9:]
        argrec = [(name, *capture_arg_meta(a)) for name, a in zip(self._capture_arg_names, kernel_args)]
        current_calls.append({
            "kernel_id": self._capture_kernel_id,
            "kernel_name": self._capture_name,
            "grid": tuple(int(g) for g in args[0:3]),
            "args": argrec,
        })
    return _orig_call(self, *args, **kwargs)


cpu_driver.CPULauncher.__init__ = capturing_init
cpu_driver.CPULauncher.__call__ = capturing_call

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

model_name = os.path.join(os.environ["HOME"], "qwen2.5-0.5b-instruct")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager").eval()

cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN, dtype=torch.bfloat16)
model.forward = torch.compile(model.forward, backend="inductor", dynamic=False)

inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"]
L = input_ids.shape[1]
print(f"prompt: {PROMPT!r}  tokens: {input_ids.tolist()[0]}  L={L}")
if L + NUM_DECODE_SAMPLES + 4 > MAX_CACHE_LEN:
    raise SystemExit(f"MAX_CACHE_LEN={MAX_CACHE_LEN} too small for L={L}")

prefill_calls = None
decode_calls_by_step = []
token_ids = [input_ids.tolist()[0]]
positions = [L]

with torch.no_grad():
    current_calls = []
    out = model.forward(input_ids=input_ids, cache_position=torch.arange(L), past_key_values=cache, use_cache=True)
    prefill_calls = current_calls
    next_id = int(out.logits[:, -1, :].argmax(-1).item())
    token_ids.append(next_id)

    for step in range(NUM_DECODE_SAMPLES):
        pos = L + step
        positions.append(pos)
        current_calls = []
        cur_input = torch.tensor([[token_ids[-1]]], dtype=torch.long)
        out = model.forward(input_ids=cur_input, cache_position=torch.tensor([pos]), past_key_values=cache,
                             use_cache=True)
        decode_calls_by_step.append(current_calls)
        next_id = int(out.logits[:, -1, :].argmax(-1).item())
        token_ids.append(next_id)

print(f"prefill: {len(prefill_calls)} launches; decode steps: {[len(c) for c in decode_calls_by_step]}")
print(f"greedy token ids from host reference run: {token_ids}")
print(f"decoded continuation: {tokenizer.decode(token_ids[L:])!r}")

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
decode0, decode1 = decode_calls_by_step[0], decode_calls_by_step[1]
assert len(decode0) == len(decode1), "decode step kernel COUNT differs -- StaticCache assumption broken"
for c0, c1 in zip(decode0, decode1):
    assert c0["kernel_id"] == c1["kernel_id"] and c0["grid"] == c1["grid"], \
        "decode step kernel identity/grid differs -- StaticCache assumption broken"
print("verified: decode0 and decode1 have identical kernel sequence + grids (safe to compile once, replay)")

written = set()
for calls in [prefill_calls, decode0, decode1]:
    for call in calls:
        for name, kind, *rest in call["args"]:
            if kind == "ptr" and (name.startswith("out_ptr") or name.startswith("in_out_ptr")):
                written.add(rest[0])  # base addr

prefill_addrs = set()
for call in prefill_calls:
    for name, kind, *rest in call["args"]:
        if kind == "ptr":
            prefill_addrs.add(rest[0])

decode0_addrs_by_pos = []  # (call_idx, arg_idx) -> (base, value)
for call in decode0:
    for name, kind, *rest in call["args"]:
        if kind == "ptr":
            decode0_addrs_by_pos.append(rest[0])

decode1_addrs_by_pos = []
for call in decode1:
    for name, kind, *rest in call["args"]:
        if kind == "ptr":
            decode1_addrs_by_pos.append(rest[0])

decode_stable = set()  # addrs appearing at the same position in decode0 == decode1
decode_varying_positions = []  # (call_idx, arg_idx, addr0, addr1, value0, value1)
idx = 0
for ci, call in enumerate(decode0):
    for ai, (name, kind, *rest) in enumerate(call["args"]):
        if kind != "ptr":
            continue
        addr0 = rest[0]
        addr1 = decode1_addrs_by_pos[idx]
        if addr0 == addr1:
            decode_stable.add(addr0)
        idx += 1

kv_state_addrs = (written & prefill_addrs) & decode_stable
weight_addrs = set()
step_input_slots = []  # list of dicts describing where to patch position/token-id per call

# Build reverse lookup: addr -> one representative (kind, value_snapshot, size, dtype)
addr_info = {}
for calls in [prefill_calls, decode0, decode1]:
    for call in calls:
        for name, kind, *rest in call["args"]:
            if kind == "ptr":
                base, off, size, dtype, val = rest
                addr_info.setdefault(base, (size, dtype, val))

all_ptr_addrs = set(addr_info.keys())
for addr in all_ptr_addrs:
    if addr in kv_state_addrs:
        continue
    if addr not in written:
        # never written -- weight (constant) or step-varying external input
        weight_addrs.add(addr)

# step-varying inputs hide inside weight_addrs (never written) but with DIFFERING
# value between decode0 and decode1 at the same call/arg position.
true_weight_addrs = set(weight_addrs)
idx = 0
for ci, call in enumerate(decode0):
    for ai, (name, kind, *rest) in enumerate(call["args"]):
        if kind != "ptr":
            continue
        addr0 = rest[0]
        val0 = rest[-1]
        addr1 = decode1_addrs_by_pos[idx]
        val1 = addr_info.get(addr1, (None, None, None))[2]
        if addr0 in weight_addrs and addr0 != addr1 and val0 != val1:
            true_weight_addrs.discard(addr0)
            step_input_slots.append({
                "call_idx": ci, "arg_idx": ai, "name": name,
                "decode0_addr": addr0, "decode0_val": val0,
                "decode1_addr": addr1, "decode1_val": val1,
                "size": addr_info[addr0][0], "dtype": addr_info[addr0][1],
            })
        idx += 1

weight_addrs = true_weight_addrs
scratch_addrs = all_ptr_addrs - kv_state_addrs - weight_addrs - {s["decode0_addr"] for s in step_input_slots}

print(f"classification: kv_state={len(kv_state_addrs)}  weight={len(weight_addrs)}  "
      f"step_input_slots={len(step_input_slots)}  scratch={len(scratch_addrs)}  "
      f"total_unique_ptr_addrs={len(all_ptr_addrs)}")
for s in step_input_slots:
    print(f"  step_input: call#{s['call_idx']} arg#{s['arg_idx']} name={s['name']!r} "
          f"decode0_val={s['decode0_val']} decode1_val={s['decode1_val']} size={s['size']} dtype={s['dtype']}")

if len(step_input_slots) == 0:
    raise SystemExit("No step-varying inputs detected -- classification is broken, aborting before writing manifest")

# ---------------------------------------------------------------------------
# Persist every unique buffer's "born" value (full whole-storage snapshot, taken
# the first time that address was ever touched during this capture session --
# see first_snapshot in capture_arg_meta). This single mechanism covers three
# different needs uniformly:
#   - weight_addrs:  this IS the real, permanent constant value.
#   - kv_state_addrs: irrelevant in practice (the driver replays prefill for
#     real, which populates these correctly) but harmless to also seed.
#   - scratch_addrs: this is exactly the value Inductor's own compiled decode
#     function had already established by the time host execution reached
#     decode0 (e.g. the self-incrementing position counter discovered during
#     capture -- read-before-written within decode0's own call sequence,
#     correctly starting at L and carried forward by the replayed kernels'
#     own internal +1 logic each iteration; no external driver input needed).
# ---------------------------------------------------------------------------
WEIGHTS_DIR = os.path.join(ENGINE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

buffer_seed_files = {}  # addr -> relative path under weights/
for addr in all_ptr_addrs:
    snap = first_snapshot.get(addr)
    if snap is None:
        continue
    fname = f"{addr}.bin"
    with open(os.path.join(WEIGHTS_DIR, fname), "wb") as f:
        # torch's own byte-reinterpretation (not numpy's, which can't represent
        # bf16) gives an exact byte dump regardless of dtype.
        f.write(bytes(snap.contiguous().view(torch.uint8).numpy()))
    buffer_seed_files[addr] = fname

print(f"Dumped {len(buffer_seed_files)} buffer snapshots ({len(weight_addrs)} weights, "
      f"{len(kv_state_addrs)} kv_state, {len(scratch_addrs)} scratch) to {WEIGHTS_DIR}")

# ---------------------------------------------------------------------------
# Phase B: compile every unique kernel to riscv64
# ---------------------------------------------------------------------------
import pickle

REBUILT_KERNELS_DIR = os.path.join(ENGINE_DIR, "rebuilt_kernels")
os.makedirs(REBUILT_KERNELS_DIR, exist_ok=True)


def write_kernel_source(kernel_id, src):
    src_lines = src.fn.raw_src
    start = next(i for i, line in enumerate(src_lines) if line.strip().startswith("@triton.jit"))
    src_text = "".join(src_lines[start:])
    path = os.path.join(REBUILT_KERNELS_DIR, f"k{kernel_id}.py")
    with open(path, "w") as f:
        f.write(src_text)
    return path


def reconstruct_kernel(src_path, name):
    ns = {"triton": triton, "tl": tl}
    try:
        from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
        ns["libdevice"] = libdevice
        ns["tl_math"] = tl_math
    except ImportError:
        pass
    with open(src_path) as f:
        exec(compile(f.read(), src_path, "exec"), ns)
    return ns[name]


kernel_manifest = {}  # kernel_id -> {arg_names, signature, constexprs, src_path, fn_name}
for kernel_id, src in captured_srcs.items():
    fn_name = src.fn.__name__
    src_path = write_kernel_source(kernel_id, src)
    kernel_manifest[kernel_id] = {
        "fn_name": fn_name,
        "src_path": src_path,
        "arg_names": list(src.fn.arg_names),
        "signature": dict(src.signature),
        "constexprs": {k: v for k, v in (dict(src.constants) if hasattr(src, "constants") else {}).items()},
    }

with open(os.path.join(ENGINE_DIR, "kernel_manifest.pkl"), "wb") as f:
    pickle.dump(kernel_manifest, f)

# ---------------------------------------------------------------------------
# Write the call-graph manifest for the driver generator
# ---------------------------------------------------------------------------
manifest = {
    "prompt": PROMPT,
    "prompt_ids": token_ids[0],
    "L": L,
    "max_cache_len": MAX_CACHE_LEN,
    "host_reference_token_ids": token_ids,
    "prefill_calls": prefill_calls,
    "decode_calls": decode0,
    "kv_state_addrs": kv_state_addrs,
    "weight_addrs": weight_addrs,
    "scratch_addrs": scratch_addrs,
    "step_input_slots": step_input_slots,
    "addr_info": addr_info,
    "buffer_seed_files": buffer_seed_files,
    "weights_dir": WEIGHTS_DIR,
}
with open(os.path.join(ENGINE_DIR, "manifest.pkl"), "wb") as f:
    pickle.dump(manifest, f)

print(f"\nWrote manifest + kernel sources to {ENGINE_DIR}")
print("Next: scripts/compile_qwen_kernels_riscv.py, then scripts/gen_qwen_driver.py")
