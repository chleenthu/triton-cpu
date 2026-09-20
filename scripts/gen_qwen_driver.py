"""Phase C: consumes manifest.pkl (build_qwen_engine.py) and compiled_kernels.pkl
(compile_qwen_kernels_riscv.py) and emits ONE driver.c that replays the real
prefill + the two captured, host-verified decode steps entirely through real
riscv64-compiled kernel .so's (LLVM built on this host), then cross-compiles
and (optionally) scp/ssh-deploys+runs the result on the board, timing TTFT/TPOT
with clock_gettime on-device.

Design, matching the buffer classification build_qwen_engine.py already did:
  - Every kernel is loaded via dlopen()/dlsym() at startup, one handle per
    kernel_id (not linked by symbol name): Inductor's generated kernel names
    are only unique WITHIN one torch.compile trace, so a prefill-trace kernel
    and a decode-trace kernel can share the exact same exported name despite
    having different real signatures (confirmed on the real Qwen capture --
    two "..._rsqrt_14" kernels, one with an i32 xnumel arg and one where
    Inductor had already folded xnumel to a constexpr). Statically linking
    multiple .so's exporting the same symbol would let the dynamic linker
    silently bind one kernel's calls to the other's code; dlopen(RTLD_LOCAL)
    keyed by kernel_id avoids that entirely.
  - One persistent malloc'd C buffer per unique captured address in
    kv_state_addrs | weight_addrs | scratch_addrs ("g_buf"), seeded ONCE at
    startup from its dumped "born" snapshot (weights/<addr>.bin) and never
    re-seeded between prefill and decode -- kv_state buffers are therefore
    genuinely written by the real prefill replay and read by decode (exactly
    like the real StaticCache), and scratch buffers are reused/overwritten
    every call exactly like Inductor's own buffer planning.
  - One small dedicated malloc'd buffer per step_input_slot ("g_step_buf"),
    rewritten with that slot's real decode0_val / decode1_val immediately
    before each decode replay -- no semantic guessing about "is this the
    position or the token id" is needed, the two real captured values are
    used directly.

Limitation: this replays exactly the 2 decode steps build_qwen_engine.py
captured and verified against the native x86 reference run (see
manifest["host_reference_token_ids"]) -- it is not open-ended generation.
Extending the position step-input further is a trivial +1 arithmetic
progression, but the token-id step-input is the argmax of that step's real
logits, and the capture does not currently label which buffer holds them;
closing that loop is future work, not attempted here.

Usage:
  python scripts/gen_qwen_driver.py [--host user@host] [--remote-dir ~/qwen_driver] [--compile-only]
Reads:  ~/qwen_triton_engine/manifest.pkl
        ~/qwen_triton_engine/compiled_kernels.pkl
Writes: ~/qwen_triton_engine/driver/driver.c
        ~/qwen_triton_engine/driver/driver          (cross-compiled riscv64 executable)
"""
import argparse
import os
import pickle
import subprocess

import triton  # noqa: F401  (imported before triton.backends.cpu.* per the rest of this project)
from triton.backends.cpu.riscv import Toolchain, deploy_and_run, ty_to_cpp, _c_literal

ENGINE_DIR = os.environ.get("QWEN_ENGINE_DIR", os.path.join(os.environ["HOME"], "qwen_triton_engine"))
DRIVER_DIR = os.path.join(ENGINE_DIR, "driver")

NUM_DECODE = 2  # matches build_qwen_engine.py's NUM_DECODE_SAMPLES -- decode0/decode1 are all we captured

TORCH_DTYPE_TO_C = {
    "torch.int64": "int64_t", "torch.int32": "int32_t", "torch.int16": "int16_t", "torch.int8": "int8_t",
    "torch.bool": "int8_t", "torch.uint8": "uint8_t",
    "torch.float32": "float", "torch.float64": "double",
    "torch.bfloat16": "__bf16", "torch.float16": "_Float16",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("TRITON_RISCV_HOST"),
                         help="user@host to scp/ssh the driver to, e.g. chlee@140.114.78.64")
    parser.add_argument("--remote-dir", default="~/qwen_driver")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--skip-weights", action="store_true",
                         help="do not copy the weight .bin files (already present in --remote-dir on the board)")
    parser.add_argument("--skip-gemm", action="store_true",
                         help="do not launch the triton_tem_* GEMM kernels (timing of the elementwise/reduction "
                              "kernels only; the model output is then meaningless)")
    parser.add_argument("--skip-name", action="append", default=[],
                         help="also skip launches whose kernel name contains this substring (repeatable)")
    parser.add_argument("--profile", action="store_true",
                         help="time every launch of the last decode step and print a per-kernel table")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds allowed for each ssh/scp step (1GB of weights + a slower full-model run)")
    args = parser.parse_args()

    with open(os.path.join(ENGINE_DIR, "manifest.pkl"), "rb") as f:
        manifest = pickle.load(f)
    with open(os.path.join(ENGINE_DIR, "compiled_kernels.pkl"), "rb") as f:
        compiled_kernels = pickle.load(f)

    addr_info = manifest["addr_info"]
    kv_state_addrs = manifest["kv_state_addrs"]
    weight_addrs = manifest["weight_addrs"]
    scratch_addrs = manifest["scratch_addrs"]
    step_input_slots = manifest["step_input_slots"]
    buffer_seed_files = manifest["buffer_seed_files"]
    weights_dir = manifest["weights_dir"]
    prefill_calls = manifest["prefill_calls"]
    decode_calls = manifest["decode_calls"]

    for s in step_input_slots:
        assert f"decode{NUM_DECODE - 1}_val" in s, \
            f"step_input_slot {s['name']!r} only has values for fewer than {NUM_DECODE} decode steps"

    generic_addrs = kv_state_addrs | weight_addrs | scratch_addrs
    addr_to_idx = {addr: i for i, addr in enumerate(sorted(generic_addrs))}
    step_input_by_pos = {(s["call_idx"], s["arg_idx"]): i for i, s in enumerate(step_input_slots)}

    def skipped(call):
        name = call["kernel_name"]
        return (args.skip_gemm and name.startswith("triton_tem")) or any(x in name for x in args.skip_name)

    used_kernel_ids = []
    seen = set()
    for call in prefill_calls + decode_calls:
        if skipped(call):
            continue
        kid = call["kernel_id"]
        if kid not in seen:
            seen.add(kid)
            used_kernel_ids.append(kid)
    missing = [kid for kid in used_kernel_ids if kid not in compiled_kernels]
    if missing:
        raise SystemExit(f"{len(missing)} kernel(s) used by the capture were never compiled "
                          f"(run scripts/compile_qwen_kernels_riscv.py first): {missing}")

    def render_args(call, call_idx):
        kid = call["kernel_id"]
        ck = compiled_kernels[kid]
        parts = []
        for arg_idx, (name, kind, *rest) in enumerate(call["args"]):
            if kind == "ptr":
                addr, offset, size, dtype, smallvals = rest
                slot_idx = step_input_by_pos.get((call_idx, arg_idx)) if call_idx is not None else None
                if slot_idx is not None:
                    buf_expr = f"g_step_buf[{slot_idx}]"
                else:
                    buf_expr = f"g_buf[{addr_to_idx[addr]}]"
                parts.append(f"(char*){buf_expr} + {offset}" if offset else buf_expr)
            else:
                (value, ) = rest
                c_type = ty_to_cpp(ck["signature"][name])
                parts.append(_c_literal(value, c_type))
        return parts

    def emit_call(call, call_idx, out):
        kid = call["kernel_id"]
        ck = compiled_kernels[kid]
        gx, gy, gz = call["grid"]
        call_prefix = ", ".join(render_args(call, call_idx))
        if call_prefix:
            call_prefix += ", "
        out.append("  {")
        out.append(f"    const int gx = {gx}, gy = {gy}, gz = {gz};")
        out.append("    #pragma omp parallel for collapse(3) schedule(static)")
        out.append("    for (int x = 0; x < gx; ++x)")
        out.append("      for (int y = 0; y < gy; ++y)")
        out.append("        for (int z = 0; z < gz; ++z)")
        out.append(f"          fn_k{kid}({call_prefix}x, y, z, gx, gy, gz);")
        out.append("  }")

    lines = [
        "// Auto-generated by scripts/gen_qwen_driver.py -- do not edit by hand.",
        "#include <dlfcn.h>",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <time.h>",
        "#ifdef _OPENMP",
        "#include <omp.h>",
        "#endif",
        "",
    ]

    # Kernels are loaded via dlopen/dlsym, one handle per kernel_id, NOT linked
    # by symbol name: Inductor's kernel-name derivation is only unique WITHIN
    # one torch.compile trace, so a prefill-trace kernel and a decode-trace
    # kernel can (and here, do) end up with the IDENTICAL exported symbol name
    # despite having different real signatures. Statically linking multiple
    # .so's that each export that same symbol name would let the dynamic
    # linker silently bind calls meant for one kernel to the other's code.
    # dlopen(..., RTLD_LOCAL) keeps each .so's symbols private to its own
    # handle, so looking the symbol up per-kernel_id is always unambiguous.
    for kid in used_kernel_ids:
        ck = compiled_kernels[kid]
        c_types = [ty_to_cpp(ck["signature"][n]) for n in ck["arg_order"]] + ["uint32_t"] * 6
        lines.append(f"typedef void (*fnty_k{kid})({', '.join(c_types)});")
        lines.append(f"static fnty_k{kid} fn_k{kid};")
    lines.append("")
    lines += [
        "static void *load_kernel_sym(const char *so_path, const char *sym) {",
        "  void *h = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);",
        '  if (!h) { fprintf(stderr, "dlopen(%s) failed: %s\\n", so_path, dlerror()); exit(1); }',
        "  dlerror();",
        "  void *fn = dlsym(h, sym);",
        "  const char *err = dlerror();",
        '  if (err) { fprintf(stderr, "dlsym(%s, %s) failed: %s\\n", so_path, sym, err); exit(1); }',
        "  return fn;",
        "}",
        "",
        "static void init_kernels(void) {",
    ]
    for kid in used_kernel_ids:
        ck = compiled_kernels[kid]
        # dlopen("bare_name.so") searches LD_LIBRARY_PATH/ldconfig paths, NOT
        # the process cwd -- an explicit "./" makes it a direct relative path
        # instead, which does resolve against cwd (the deploy dir on the board).
        so_name = "./" + os.path.basename(ck["so_path"])
        lines.append(f'  fn_k{kid} = (fnty_k{kid})load_kernel_sym("{so_name}", "{ck["real_name"]}");')
    lines.append("}")
    lines.append("")

    lines.append(f"static void *g_buf[{max(len(addr_to_idx), 1)}];")
    lines.append(f"static void *g_step_buf[{max(len(step_input_slots), 1)}];")
    lines.append("")
    lines += [
        "static void *load_buffer(const char *path, size_t size) {",
        "  void *p = malloc(size ? size : 1);",
        '  if (!p) { fprintf(stderr, "OOM allocating %zu bytes for %s\\n", size, path); exit(1); }',
        '  FILE *f = fopen(path, "rb");',
        '  if (!f) { fprintf(stderr, "cannot open %s\\n", path); exit(1); }',
        "  size_t got = fread(p, 1, size, f);",
        "  fclose(f);",
        '  if (got != size) { fprintf(stderr, "short read on %s: got %zu want %zu\\n", path, got, size); exit(1); }',
        "  return p;",
        "}",
        "",
        "static double ms_between(struct timespec a, struct timespec b) {",
        "  return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;",
        "}",
        "",
        "static void init_buffers(void) {",
    ]
    for addr, idx in addr_to_idx.items():
        size, _dtype, _val = addr_info[addr]
        fname = buffer_seed_files[addr]
        lines.append(f'  g_buf[{idx}] = load_buffer("{fname}", {size}UL);')
    for i, slot in enumerate(step_input_slots):
        lines.append(f"  g_step_buf[{i}] = malloc({slot['size']}UL);")
    lines.append("}")
    lines.append("")

    if args.profile:
        lines.append(f"static double g_prof[{max(len(decode_calls), 1)}];")
        lines.append("static const char *g_prof_name[] = {")
        for call in decode_calls:
            lines.append(f'  "{call["kernel_name"]} grid={call["grid"]}",')
        lines.append("};")
        lines.append("")
    lines.append("int main(void) {")
    lines.append("  init_kernels();")
    lines.append("  init_buffers();")
    lines.append(f"  struct timespec t_start, t_prefill, t_decode[{NUM_DECODE}];")
    lines.append("  clock_gettime(CLOCK_MONOTONIC, &t_start);")
    lines.append("")
    lines.append("  // ---- prefill ----")
    for call in prefill_calls:
        if not skipped(call):
            emit_call(call, None, lines)
    lines.append("  clock_gettime(CLOCK_MONOTONIC, &t_prefill);")
    lines.append("")

    for step in range(NUM_DECODE):
        lines.append(f"  // ---- decode step {step} ----")
        for slot_idx, slot in enumerate(step_input_slots):
            val = slot[f"decode{step}_val"][0]
            c_type = TORCH_DTYPE_TO_C.get(slot["dtype"])
            if c_type is None:
                raise SystemExit(f"no C type mapping for step_input dtype {slot['dtype']!r} "
                                  f"(slot {slot['name']!r}) -- add it to TORCH_DTYPE_TO_C")
            lit = _c_literal(val, c_type)
            lines.append(f"  *({c_type}*)g_step_buf[{slot_idx}] = {lit};")
        for call_idx, call in enumerate(decode_calls):
            if skipped(call):
                continue
            prof = args.profile and step == NUM_DECODE - 1
            if prof:
                lines.append("  { struct timespec pa, pb; clock_gettime(CLOCK_MONOTONIC, &pa);")
            emit_call(call, call_idx, lines)
            if prof:
                lines.append(f"  clock_gettime(CLOCK_MONOTONIC, &pb); g_prof[{call_idx}] = ms_between(pa, pb); }}")
        lines.append(f"  clock_gettime(CLOCK_MONOTONIC, &t_decode[{step}]);")
        lines.append("")

    lines += [
        '  printf("TTFT: %.3f ms  (prefill %.3f ms + decode0 %.3f ms)\\n",',
        "         ms_between(t_start, t_decode[0]), ms_between(t_start, t_prefill),",
        "         ms_between(t_prefill, t_decode[0]));",
    ]
    if NUM_DECODE > 1:
        lines.append('  printf("TPOT: %.3f ms  (decode1 - decode0)\\n", ms_between(t_decode[0], t_decode[1]));')
    if args.profile:
        lines.append(f"  for (int i = 0; i < {len(decode_calls)}; ++i) printf(\"PROF %9.3f ms  %s\\n\", g_prof[i], g_prof_name[i]);")
    lines += [
        '  puts("PASS");',
        "  return 0;",
        "}",
        "",
    ]

    os.makedirs(DRIVER_DIR, exist_ok=True)
    driver_c_path = os.path.join(DRIVER_DIR, "driver.c")
    with open(driver_c_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {driver_c_path}  ({len(used_kernel_ids)} kernels, {len(addr_to_idx)} persistent buffers, "
          f"{len(step_input_slots)} step inputs)")

    so_paths = [compiled_kernels[kid]["so_path"] for kid in used_kernel_ids]
    driver_exe = os.path.join(DRIVER_DIR, "driver")
    toolchain = Toolchain.from_env()
    # Kernel .so's are NOT link inputs -- they're dlopen()'d at runtime (see
    # above), so driver.c only needs libdl (folded into libc on newer glibc,
    # but -ldl is harmless there and required on older ones).
    extra_args = ["-fopenmp", "-ldl"]
    if toolchain.sysroot and os.path.isfile(os.path.join(toolchain.sysroot, "usr", "lib", "libomp.a")):
        extra_args += ["-Wl,-Bstatic", "-lomp", "-Wl,-Bdynamic"]
    cmd = toolchain.compile_command([driver_c_path], driver_exe, extra_args=extra_args)
    print("compiling driver:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"built {driver_exe}")

    print(f"\nhost reference token ids (prompt + 2 greedy decode steps): "
          f"{manifest['host_reference_token_ids']}")
    print("(the driver above only times prefill+2 decode steps; it does not verify logits on-device)")

    if args.compile_only:
        return
    if not args.host:
        raise SystemExit("No --host given (and TRITON_RISCV_HOST unset); pass --compile-only to skip deploy+run")

    bin_paths = [os.path.join(weights_dir, buffer_seed_files[addr]) for addr in addr_to_idx]
    local_files = [driver_exe, *so_paths, *([] if args.skip_weights else bin_paths)]
    print(f"\ndeploying {len(local_files)} files to {args.host}:{args.remote_dir} ...")
    result = deploy_and_run(local_files, args.host, os.path.basename(driver_exe), remote_dir=args.remote_dir,
                             timeout=args.timeout)
    print(result.stdout, end="")
    print(result.stderr, end="")
    if result.returncode != 0 or "PASS" not in result.stdout:
        raise SystemExit(f"driver run failed (exit code {result.returncode})")


if __name__ == "__main__":
    main()
