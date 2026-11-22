# riscv64 (rvv_*_elf.py-style) replay of the captured vLLM unified-attention
# launches: cross-compile each kernel to a standalone ELF + .so, scp to the
# board, run it there and check the outputs against what vLLM produced.
#
#   ./run_riscv.sh                       # DEFAULT_CASES
#   ./run_riscv.sh --case all            # every captured call
#   ./run_riscv.sh --compile-only        # build the ELFs, don't run them
#
# Kernel bodies, DEFAULT_CASES and the captured/*.pt loading are shared with
# unified_attention_repro.py (the x86 / native flow).
#
# Differences from the native flow:
#   * every buffer is baked into the generated C runner as an initializer
#     (flattened); the KV caches are passed as their captured storage, which
#     is exactly what the strided view indexes from offset 0.
#   * None arguments (segm_* in 2D) become constexpr None.
#   * reduce_segments uses vLLM's captured segm_* inputs (no chaining with the
#     3D kernel, which ran in a separate ELF).
#   * outputs are checked with atol (fp16 output, fp32 softmax on CPU), not
#     bit-exactness. NaN in the expected buffer (never-written segm slots)
#     is skipped by the runner's check.
import argparse
import glob
import os

import torch
import triton

from triton.backends.cpu.riscv import compile_deploy_and_run, compile_kernel_to_elf

import unified_attention_repro as repro

HERE = os.path.dirname(os.path.abspath(__file__))
RISCV_HOST = os.environ.get("TRITON_RISCV_HOST", "chlee@140.114.78.64")
RISCV_REMOTE_DIR = "~/triton-riscv-elf"
ATOL = 1e-2

_TORCH_TO_TT = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32",
                torch.int32: "i32", torch.int64: "i64"}


def build_launch(name):
    rec = torch.load(os.path.join(repro.CAPTURED, name + ".pt"), weights_only=False)
    fn = repro.KERNELS[rec["name"]]
    annotated = {p.name: p.annotation for p in fn.params}

    arguments, constexprs, signature, expected = {}, {}, {}, {}
    for k in fn.arg_names:
        if k not in rec["kwargs"]:
            if k in ("segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr"):
                constexprs[k] = None  # default=None in the kernel
                continue
            raise KeyError(f"{name}: no captured value for {k}")
        v = rec["kwargs"][k]
        if isinstance(v, dict) and "storage" in v:  # KV cache: pass the storage
            v = v["storage"]
        if annotated.get(k) == "constexpr" or v is None:
            constexprs[k] = v
        elif isinstance(v, torch.Tensor):
            arguments[k] = v.contiguous().flatten()
            signature[k] = "*" + _TORCH_TO_TT[v.dtype]
        else:
            arguments[k] = v
            if annotated.get(k) in ("i32", "i64"):
                signature[k] = annotated[k]
            elif isinstance(v, float):
                signature[k] = "fp32"
    for k, ref in rec["after"].items():
        expected[k] = ref.flatten().tolist()
    return rec, fn, arguments, constexprs, signature, expected


def run_case(name, compile_only):
    rec, fn, arguments, constexprs, signature, expected = build_launch(name)
    print(f"== {name}  grid={rec['grid']}", flush=True)
    # one dir per case: the kernel .so is named after the kernel, so the 2D and
    # 3D cases of kernel_unified_attention would overwrite each other's .so
    out = os.path.join(HERE, "artifacts", "riscv", name, f"{name}.elf")
    kw = dict(constexprs=constexprs, signature=signature, expected=expected, atol=ATOL)
    if compile_only:
        exe, so = compile_kernel_to_elf(fn, arguments, rec["grid"], out, **kw)
        print(f"   built {exe} + {so.name}")
        return True
    result = compile_deploy_and_run(fn, arguments, rec["grid"], out, RISCV_HOST,
                                    remote_dir=f"{RISCV_REMOTE_DIR}/{name}", **kw)
    print(result.stdout, end="")
    print(result.stderr, end="")
    return result.returncode == 0 and "PASS" in result.stdout


def main():
    ap = argparse.ArgumentParser()
    # Default: only the real 128-token prefill (2D). Pass --case for others.
    ap.add_argument("--case", nargs="*", default=["kernel_unified_attention_2d_q128_seqlen128"],
                    help="captured case names, or 'all'")
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()

    triton.runtime.driver.set_active_to_cpu()
    allcases = sorted(os.path.basename(p)[:-3]
                      for p in glob.glob(os.path.join(repro.CAPTURED, "*.pt")))
    cases = allcases if a.case == ["all"] else a.case

    ok = True
    for c in cases:
        ok &= run_case(c, a.compile_only)
    print("riscv64:", "ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
