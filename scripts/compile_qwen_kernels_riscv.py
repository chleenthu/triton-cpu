"""Phase B (the real one): build_qwen_engine.py's docstring claims Phase B
"reconstruct[s] + compile[s] to riscv64", but its actual code only writes
kernel *sources* + signatures to kernel_manifest.pkl -- it never calls
compile_kernel_to_so. This script is that missing step: reconstruct every
kernel named in kernel_manifest.pkl and cross-compile each one to a real
riscv64 .so via riscv.py's compile_kernel_to_so, using the LLVM/clang
toolchain built on this host (CC=.../install/bin/clang).

TRITON_CPU_TARGET is deliberately left unset here so it defaults to "riscv64"
(third_party/cpu/backend/driver.py:49) -- the real cross-compile path, not
the native-x86 one build_qwen_engine.py's Phase A used.

Usage: python scripts/compile_qwen_kernels_riscv.py [--phase prefill|decode] [KERNEL_NAME ...]
  With names, only kernels whose fn_name is one of them are recompiled; every
  other entry of an existing compiled_kernels.pkl (and its .so) is kept.
  Prefill and decode are separate traces whose kernels can share a name;
  --phase keeps only the kernels launched in that pass (per manifest.pkl).
Reads:  ~/qwen_triton_engine/kernel_manifest.pkl   (from build_qwen_engine.py)
Writes: ~/qwen_triton_engine/rebuilt_kernels/k<id>.so   (one per kernel)
        ~/qwen_triton_engine/compiled_kernels.pkl
Next:   scripts/gen_qwen_driver.py
"""
import os

os.environ["TRITON_DEFAULT_BACKEND"] = "cpu"

import argparse
import pickle
import traceback

import triton
import triton.language as tl

triton.runtime.driver.set_active_to_cpu()

from triton.backends.cpu.riscv import compile_kernel_to_so

ENGINE_DIR = os.environ.get("QWEN_ENGINE_DIR", os.path.join(os.environ["HOME"], "qwen_triton_engine"))
REBUILT_KERNELS_DIR = os.path.join(ENGINE_DIR, "rebuilt_kernels")
os.makedirs(REBUILT_KERNELS_DIR, exist_ok=True)

with open(os.path.join(ENGINE_DIR, "kernel_manifest.pkl"), "rb") as f:
    kernel_manifest = pickle.load(f)


def reconstruct_kernel(src_path, name):
    # Mirrors build_qwen_engine.py's own reconstruct_kernel: @triton.jit's
    # JITFunction.__init__ calls inspect.getsourcelines(fn), which needs a
    # real file on disk (linecache can't resolve a synthetic exec filename),
    # so src_path must already exist on disk (it does -- build_qwen_engine.py
    # wrote it).
    ns = {"triton": triton, "tl": tl}
    try:
        from torch._inductor.runtime import triton_helpers
        from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
        ns["triton_helpers"] = triton_helpers
        ns["libdevice"] = libdevice
        ns["tl_math"] = tl_math
    except ImportError:
        pass
    with open(src_path) as f:
        exec(compile(f.read(), src_path, "exec"), ns)
    return ns[name]


def constexprs_by_name(km):
    # build_qwen_engine.py's captured src.constants (this dict, verbatim) is
    # keyed by a 1-tuple of the arg's positional INDEX (e.g. {(3,): 128} for
    # XBLOCK at arg_names[3]) -- a triton-internal convention, not a capture
    # bug. compile_kernel_to_so's make_standalone_source looks constexprs up
    # by NAME, so translate index -> name here.
    arg_names = km["arg_names"]
    out = {}
    for key, val in km["constexprs"].items():
        idx = key[0] if isinstance(key, tuple) else key
        out[arg_names[idx]] = val
    return out


parser = argparse.ArgumentParser()
parser.add_argument("names", nargs="*", metavar="KERNEL_NAME")
parser.add_argument("--phase", choices=["prefill", "decode"])
cli = parser.parse_args()
only_names = set(cli.names)
phase_ids = None
if cli.phase:
    with open(os.path.join(ENGINE_DIR, "manifest.pkl"), "rb") as f:
        phase_ids = {c["kernel_id"] for c in pickle.load(f)[f"{cli.phase}_calls"]}

compiled_kernels = {}
compiled_pkl = os.path.join(ENGINE_DIR, "compiled_kernels.pkl")
if only_names or phase_ids is not None:
    with open(compiled_pkl, "rb") as f:
        compiled_kernels = pickle.load(f)
    unknown = only_names - {km["fn_name"] for km in kernel_manifest.values()}
    if unknown:
        raise SystemExit(f"no kernel named {sorted(unknown)} in kernel_manifest.pkl")
failures = []
for kernel_id, km in kernel_manifest.items():
    if only_names and km["fn_name"] not in only_names:
        continue
    if phase_ids is not None and kernel_id not in phase_ids:
        continue
    print(f"=== compiling k{kernel_id} ({km['fn_name']}) -> riscv64 ===")
    try:
        kernel = reconstruct_kernel(km["src_path"], km["fn_name"])
        # Every non-constexpr arg's real triton type is already known from the
        # capture (km["signature"]), so compile_kernel_to_so needs no real
        # tensor values -- passing signature= for every arg (constexprs are
        # matched by name first regardless) lets `arguments={}` satisfy
        # make_standalone_source's "every arg_name must resolve somewhere"
        # requirement without ever touching real data.
        signature = {n: t for n, t in km["signature"].items() if t != "constexpr"}
        compiled, so_bytes = compile_kernel_to_so(kernel, arguments={}, constexprs=constexprs_by_name(km),
                                                    signature=signature)
    except Exception:
        traceback.print_exc()
        failures.append(kernel_id)
        print(f"  -> FAILED")
        continue

    so_path = os.path.join(REBUILT_KERNELS_DIR, f"k{kernel_id}.so")
    with open(so_path, "wb") as f:
        f.write(so_bytes)

    arg_order = [n for n in km["arg_names"] if km["signature"].get(n) != "constexpr"]
    compiled_kernels[kernel_id] = {
        "so_path": so_path,
        "real_name": compiled.metadata.name,
        "arg_order": arg_order,
        "signature": {n: km["signature"][n] for n in arg_order},
    }
    #print(f"  -> {compiled.metadata.name}  ({so_path})")

with open(compiled_pkl, "wb") as f:
    pickle.dump(compiled_kernels, f)

if only_names or phase_ids is not None:
    print(f"\nrecompiled only {sorted(only_names) or 'all'} ({cli.phase or 'both phases'}); kept the other {len(compiled_kernels)}/{len(kernel_manifest)} "
          f"entries' .so under {REBUILT_KERNELS_DIR}")
else:
    print(f"\n{len(compiled_kernels)}/{len(kernel_manifest)} kernels compiled to riscv64 .so under "
          f"{REBUILT_KERNELS_DIR}")
if failures:
    print(f"FAILED kernel ids: {failures}")
print("Next: scripts/gen_qwen_driver.py")
