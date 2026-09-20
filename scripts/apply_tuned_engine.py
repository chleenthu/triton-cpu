"""Rebuild the captured Qwen engine with the board-tuned GEMM tiles (scripts/tune_gemm_on_board.py).

Reads   ~/qwen_triton_engine (manifest.pkl, kernel_manifest.pkl, compiled_kernels.pkl, rebuilt_kernels/)
Writes  ~/qwen_triton_engine_tuned  (same layout; the original engine is left untouched)

For every GEMM template kernel whose tuned tile (gemm_table.json, keyed MxNxKxdtype) differs from the
tile baked into its source, rewrite BLOCK_M/N/K, recompile that kernel to riscv64, and recompute its
launch grid in the captured prefill/decode calls (grid.x = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)).
All other kernels, the weights and the buffer plan are reused as they are. Then run

    QWEN_ENGINE_DIR=~/qwen_triton_engine_tuned python scripts/gen_qwen_driver.py --compile-only

to build the tuned driver.
"""
import argparse
import importlib.util
import json
import os
import pickle
import re
import shutil
import sys
import traceback

os.environ.setdefault("TRITON_DEFAULT_BACKEND", "cpu")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

triton.runtime.driver.set_active_to_cpu()
from triton.backends.cpu.riscv import compile_kernel_to_so  # noqa: E402

HOME = os.environ["HOME"]


def reconstruct_kernel(src_path, name):
    ns = {"triton": triton, "tl": tl}
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    ns.update(triton_helpers=triton_helpers, libdevice=libdevice, tl_math=tl_math)
    exec(compile(open(src_path).read(), src_path, "exec"), ns)
    return ns[name]


def constexprs_by_name(km):
    out = {}
    for key, val in km["constexprs"].items():
        out[km["arg_names"][key[0] if isinstance(key, tuple) else key]] = val
    return out


def cdiv(a, b):
    return -(-a // b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-dir", default=os.path.join(HOME, "qwen_triton_engine"))
    ap.add_argument("--out-dir", default=os.path.join(HOME, "qwen_triton_engine_tuned"))
    ap.add_argument("--table", default="artifacts/riscv/tune/qwen_forward/gemm_table.json")
    args = ap.parse_args()

    src_dir, out_dir = args.engine_dir, args.out_dir
    table = json.load(open(args.table))
    km_all = pickle.load(open(os.path.join(src_dir, "kernel_manifest.pkl"), "rb"))
    ck_all = pickle.load(open(os.path.join(src_dir, "compiled_kernels.pkl"), "rb"))
    manifest = pickle.load(open(os.path.join(src_dir, "manifest.pkl"), "rb"))

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(os.path.join(out_dir, "rebuilt_kernels"))
    old_k, new_k = os.path.join(src_dir, "rebuilt_kernels"), os.path.join(out_dir, "rebuilt_kernels")
    for f in os.listdir(old_k):
        shutil.copy2(os.path.join(old_k, f), new_k)
    retarget = lambda path: os.path.join(new_k, os.path.basename(path))  # noqa: E731
    for km in km_all.values():
        km["src_path"] = retarget(km["src_path"])
    for ck in ck_all.values():
        ck["so_path"] = retarget(ck["so_path"])

    calls = manifest["prefill_calls"] + manifest["decode_calls"]
    changed = []
    for kid, km in km_all.items():
        if not km["fn_name"].startswith("triton_tem"):
            continue
        src = open(km["src_path"]).read()
        dims = [int(re.search(rf"^\s+{d} = (\d+)\s*$", src, re.M).group(1)) for d in "MNK"]
        tile = {k: int(re.search(rf"\b{k} : tl\.constexpr = (\d+)", src).group(1))
                for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K")}
        ref = next(a for a in ("arg_A", "arg_B", *km["arg_names"]) if a in km["signature"])
        dtype = km["signature"][ref].lstrip("*")
        key = "x".join(map(str, dims)) + f"x{dtype}"
        hit = table.get(key)
        if not hit or all(hit[k] == tile[k] for k in tile):
            print(f"  keep   {km['fn_name'][:60]:60s} {key} tile {tile['BLOCK_M']}/{tile['BLOCK_N']}/{tile['BLOCK_K']}")
            continue
        new = {k: hit[k] for k in tile}
        # the captured grid must match the old tile, or the grid formula does not apply to this kernel
        mine = [c for c in calls if c["kernel_id"] == kid]
        old_x = cdiv(dims[0], tile["BLOCK_M"]) * cdiv(dims[1], tile["BLOCK_N"])
        if any(c["grid"][0] != old_x for c in mine):
            print(f"  SKIP   {km['fn_name'][:60]}: captured grid does not match the tile formula")
            continue
        for k, v in new.items():
            src = re.sub(rf"(\b{k} : tl\.constexpr = )\d+", lambda m_, v=v: m_.group(1) + str(v), src, count=1)
        even = "True" if dims[2] % new["BLOCK_K"] == 0 else "False"
        src = re.sub(r"(EVEN_K : tl\.constexpr = )(True|False)", lambda m_: m_.group(1) + even, src, count=1)
        open(km["src_path"], "w").write(src)
        try:
            kernel = reconstruct_kernel(km["src_path"], km["fn_name"])
            sig = {n: t for n, t in km["signature"].items() if t != "constexpr"}
            compiled, so_bytes = compile_kernel_to_so(kernel, arguments={}, constexprs=constexprs_by_name(km),
                                                       signature=sig)
        except Exception:
            traceback.print_exc()
            sys.exit(f"compile failed for {km['fn_name']}")
        open(ck_all[kid]["so_path"], "wb").write(so_bytes)
        ck_all[kid]["real_name"] = compiled.metadata.name
        new_x = cdiv(dims[0], new["BLOCK_M"]) * cdiv(dims[1], new["BLOCK_N"])
        for c in mine:
            c["grid"] = (new_x,) + tuple(c["grid"][1:])
        changed.append((km["fn_name"], key, tile, new, len(mine)))
        print(f"  RETILE {km['fn_name'][:60]:60s} {key} {tile['BLOCK_M']}/{tile['BLOCK_N']}/{tile['BLOCK_K']} -> "
              f"{new['BLOCK_M']}/{new['BLOCK_N']}/{new['BLOCK_K']}  ({len(mine)} launches, grid.x {old_x}->{new_x})")

    for name, obj in (("kernel_manifest.pkl", km_all), ("compiled_kernels.pkl", ck_all), ("manifest.pkl", manifest)):
        pickle.dump(obj, open(os.path.join(out_dir, name), "wb"))
    print(f"\n{len(changed)} kernel(s) retiled -> {out_dir}")
    print(f"next: QWEN_ENGINE_DIR={out_dir} python scripts/gen_qwen_driver.py --compile-only")


if __name__ == "__main__":
    main()
