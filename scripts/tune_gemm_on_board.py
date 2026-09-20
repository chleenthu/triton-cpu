"""Tune the GEMM tile sizes of a *real* model on the riscv64 board.

    python scripts/tune_gemm_on_board.py scripts/qwen_forward.py

The only thing you change between models is the model script's file name. The script is run
as-is (Dynamo -> FX graph -> Inductor with cpu_backend="triton"); this flow hooks Inductor's
kernel compilation, so it sees exactly the GEMM template kernels (triton_tem_*: mm / addmm /
bmm with their fused epilogues, real M/N/K/dtype/strides) that the model produces:

  A. capture  run the model script with kernel compilation/launch stubbed out (no riscv code
              is built or run here), recording every triton_tem_* kernel: source, buffer
              sizes, grid.
  B. sweep    for each distinct (M, N, K, dtype): rewrite BLOCK_M/N/K in the real kernel
              source, cross-compile every candidate to a riscv64 ELF, copy them to the board
              in one scp, run them there (timed loop, output checksum) and keep the fastest.
  C. table    write gemm_table.json. Use it with the patched Inductor
              (scripts/patch_inductor_cpu_triton_gemm.py):

                  TRITON_CPU_GEMM_TABLE=<gemm_table.json> python <model script>

Verification: candidates whose output checksum deviates from the median are discarded
(wrong-result tiles); the tile Inductor originally picked on x86 is always included as the
baseline, so the report shows the speedup over it.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import re
import runpy
import shutil
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_HOST = "chlee@140.114.78.64"
TILE_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_K")
DTYPE_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "int32": "i32", "int64": "i64",
               "int8": "i8", "float64": "fp64"}
C_TYPE = {"bf16": "uint16_t", "fp16": "_Float16", "fp32": "float", "fp64": "double", "i32": "int", "i64": "long",
          "i8": "signed char"}  # bf16 buffers are raw bit patterns: avoids double<->bf16 libcalls the toolchain lacks


# --------------------------------------------------------------------------------------
# Phase A: capture the real GEMM kernels of the model script
# --------------------------------------------------------------------------------------
class _KernelStub:
    """Stands in for Inductor's compiled kernel: records the launch instead of running it."""

    def __init__(self, store, name, source):
        # Prefill and decode are separate graphs and Inductor reuses kernel names across them, so
        # key by content: identical source is one kernel, same name with different source is not.
        digest = hashlib.md5(source.encode()).hexdigest()[:8]
        rec = store.setdefault(f"{name}@{digest}", {"name": name, "source": source, "args": None,
                                                       "grid": None, "launches": 0})
        self.rec = rec

    def run(self, *args, **kwargs):
        rec = self.rec
        rec["launches"] += 1
        if rec["args"] is None:
            tensors = [a for a in args if hasattr(a, "numel") and hasattr(a, "dtype")]
            ints = [a for a in args if isinstance(a, int)]
            rec["args"] = [(int(t.numel()), DTYPE_SHORT.get(str(t.dtype).removeprefix("torch."), "?"))
                           for t in tensors]
            rec["grid"] = tuple(ints[-3:]) if len(ints) >= 3 else (1, 1, 1)
            rec["nscalars"] = len(ints) - 3

    __call__ = run

    def precompile(self, *a, **k):
        pass


def capture(model_script, script_args, cache_dir):
    os.environ.setdefault("TRITON_DEFAULT_BACKEND", "cpu")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    import triton
    triton.runtime.driver.set_active_to_cpu()
    import torch._inductor.config as ic
    ic.cpu_backend = "triton"
    ic.max_autotune = False
    ic.max_autotune_gemm = False
    from torch._inductor.async_compile import AsyncCompile

    store = {}
    AsyncCompile.triton = lambda self, kernel_name, source_code, device_str="cuda": _KernelStub(
        store, kernel_name, source_code)

    sys.argv = [str(model_script), *script_args]
    try:
        runpy.run_path(str(model_script), run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:  # the stubbed run produces garbage tensors; keep whatever was captured
        print(f"[capture] model script stopped with {type(e).__name__}: {str(e)[:200]}")
    return store


# --------------------------------------------------------------------------------------
# Phase B: sweep tile configs on the board
# --------------------------------------------------------------------------------------
def analyze(rec):
    """Pull (M, N, K, dtype, arg names, current tile, warps/stages) out of a template kernel."""
    name = rec["name"]
    src = rec["source"]
    if not name.startswith("triton_tem_") or rec["args"] is None or rec.get("nscalars", 0) != 0:
        return None
    dims = {d: re.search(rf"^\s+{d} = (\d+)\s*$", src, re.M) for d in "MNK"}
    tile = {k: re.search(rf"\b{k} : tl\.constexpr = (\d+)", src) for k in TILE_KEYS}
    defn = re.search(rf"def {re.escape(name)}\(([^)]*)\)", src)
    if not defn or not all(dims.values()) or not all(tile.values()):
        return None
    arg_names = [a.strip() for a in defn.group(1).split(",")]
    if len(arg_names) != len(rec["args"]):
        return None
    # A can be fused away into a prologue (no arg_A); fall back to B, then to the first buffer.
    ref = next((a for a in ("arg_A", "arg_B") if a in arg_names), arg_names[0])
    dtype = rec["args"][arg_names.index(ref)][1]
    m, n, k = (int(dims[d].group(1)) for d in "MNK")
    group_m = re.search(r"\bGROUP_M : tl\.constexpr = (\d+)", src)
    return {
        "key": f"{m}x{n}x{k}x{dtype}", "M": m, "N": n, "K": k, "dtype": dtype, "arg_names": arg_names,
        "tile": {k_: int(v.group(1)) for k_, v in tile.items()},
        "group_m": int(group_m.group(1)) if group_m else None,
        "num_warps": int((re.search(r"num_warps=(\d+)", src) or [0, 1])[1]),
        "num_stages": int((re.search(r"num_stages=(\d+)", src) or [0, 1])[1]),
    }


def with_tile(src, cfg, k):
    for key, val in cfg.items():
        src = re.sub(rf"(\b{key} : tl\.constexpr = )\d+", lambda m_, v=val: m_.group(1) + str(v), src, count=1)
    even = "True" if k % cfg["BLOCK_K"] == 0 else "False"
    src = re.sub(r"(EVEN_K : tl\.constexpr = )(True|False)", lambda m_: m_.group(1) + even, src, count=1)
    # Drop Inductor's autotuning decorator; the tile is now fixed in the source.
    return re.sub(r"@triton_heuristics\.template\(.*?\n\)\s*@triton\.jit", "@triton.jit", src, count=1, flags=re.S)


def candidate_tiles(info, args):
    p2 = lambda x: 1 << max(int(x) - 1, 0).bit_length()  # noqa: E731
    elem = {"bf16": 2, "fp16": 2, "fp32": 4}.get(info["dtype"], 4)
    cands = []
    for bm, bn, bk in itertools.product(args.block_m, args.block_n, args.block_k):
        if bm > max(16, p2(info["M"])) or bn > max(16, p2(info["N"])) or bk > max(16, p2(info["K"])):
            continue
        if (bm * bk + bk * bn) * elem > args.l1_bytes or bm * bn * 4 > args.acc_bytes:
            continue
        cands.append({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})
    base = dict(info["tile"])
    cands = [c for c in cands if c != base]
    cands.insert(0, base)  # baseline (the x86-chosen tile) is always first
    return cands


def load_jit(path, kname):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod  # triton.jit reads the source through the module
    spec.loader.exec_module(mod)
    return getattr(mod, kname)


def make_runner(kname, sig, grid, out_indices):
    """C runner: fills the (static) buffers with a small deterministic pattern, times the grid loop
    (same OpenMP scheme as triton.backends.cpu.riscv.generate_runner) and prints an output checksum."""
    decls, protos, call = [], [], []
    for i, (elem_ty, numel) in enumerate(sig):
        c_ty = C_TYPE.get(elem_ty, "float")
        decls.append(f"static {c_ty} buf{i}[{max(numel, 1)}];")
        protos.append("void*")
        call.append(f"buf{i}")
    def fill_line(i, t, n):
        val = "(float)((((long)j * 2654435761u) >> 13) % 9) * 0.125f - 0.5f"
        if t == "bf16":
            return f"  for (long j = 0; j < {max(n, 1)}; ++j) buf{i}[j] = f2bf({val});"
        return f"  for (long j = 0; j < {max(n, 1)}; ++j) buf{i}[j] = ({C_TYPE[t]})({val});"

    fill = [fill_line(i, t, n) for i, (t, n) in enumerate(sig) if t in ("bf16", "fp16", "fp32", "fp64")]
    gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
    loop = [
        "#ifdef _OPENMP", "#pragma omp for collapse(3) schedule(static)", "#endif",
        f"      for (int x = 0; x < {gx}; ++x)", f"        for (int y = 0; y < {gy}; ++y)",
        f"          for (int z = 0; z < {gz}; ++z)",
        f"            {kname}({', '.join(call)}, x, y, z, {gx}, {gy}, {gz});",
    ]
    checks = [f"  for (long j = 0; j < {max(sig[i][1], 1)}; ++j) cs += " +
              (f"bf2f(buf{i}[j]);" if sig[i][0] == "bf16" else f"(double)buf{i}[j];") for i in out_indices]
    lines = [
        "#include <stdint.h>", "#include <stdio.h>", "#include <stdlib.h>", "#include <time.h>",
        "#include <string.h>", "#ifdef _OPENMP", "#include <omp.h>", "#endif", "",
        "static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }",
        "static double bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }", "",
        f"extern void {kname}({', '.join(protos + ['uint32_t'] * 6)});", "", *decls, "",
        "int main(void) {", *fill,
        "  long iters = 1, warmup = 0; const char *e;",
        '  if ((e = getenv("TRITON_BENCH_ITERS"))) iters = atol(e);',
        '  if ((e = getenv("TRITON_BENCH_WARMUP"))) warmup = atol(e);',
        "  if (iters < 1) iters = 1; if (warmup < 0) warmup = 0;",
        "#ifdef _OPENMP", "  #pragma omp parallel", "#endif",
        "  { for (long it = 0; it < warmup; ++it) {", *loop, "    } }",
        "  struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);",
        "#ifdef _OPENMP", "  #pragma omp parallel", "#endif",
        "  { for (long it = 0; it < iters; ++it) {", *loop, "    } }",
        "  clock_gettime(CLOCK_MONOTONIC, &t1);",
        "  double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec);",
        '  printf("Time: %.4f ms (mean of %ld iters)\\n", ns / 1e6 / iters, iters);',
        "  double cs = 0;", *checks,
        '  printf("Checksum: %.9e\\n", cs);', '  puts("DONE");', "  return 0;", "}", "",
    ]
    return "\n".join(lines)


def build_variant(info, rec, cfg, workdir, args):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.cpu.riscv import Toolchain, build_standalone_executable
    from triton.compiler import ASTSource

    name = "M{BLOCK_M}_N{BLOCK_N}_K{BLOCK_K}".format(**cfg)
    vdir = workdir / name
    vdir.mkdir(parents=True, exist_ok=True)
    kname = rec["name"]
    src_path = vdir / f"{kname}_{name}.py"
    src_path.write_text(with_tile(rec["source"], cfg, info["K"]))
    t0 = time.time()
    fn = load_jit(src_path, kname)
    sig = {a: f"*{rec['args'][i][1]}" for i, a in enumerate(info["arg_names"])}
    target = GPUTarget("cpu", 0, 0)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": info["num_warps"], "num_stages": info["num_stages"]})
    compiled = triton.compile(ASTSource(fn=fn, signature=sig, constexprs={}), target=target,
                              options=options.__dict__)
    so_bytes = compiled.asm[backend.binary_ext]
    t_compile = time.time() - t0
    grid = (-(-info["M"] // cfg["BLOCK_M"]) * -(-info["N"] // cfg["BLOCK_N"]),) + tuple(rec["grid"][1:])
    out_idx = [i for i, a in enumerate(info["arg_names"]) if a.startswith(("out_ptr", "in_out_ptr"))]
    runner = make_runner(compiled.metadata.name, [(rec["args"][i][1], rec["args"][i][0])
                                                  for i in range(len(info["arg_names"]))], grid, out_idx)
    t1 = time.time()
    build_standalone_executable(compiled.metadata.name, so_bytes, runner, vdir / "k.elf",
                                toolchain=Toolchain.from_env())
    print(f"    {name}: kernel compile {t_compile:.0f}s, runner build {time.time() - t1:.0f}s", flush=True)
    return name


def run_on_board(shape_dir, args):
    remote = f"{args.remote_dir}/{shape_dir.name}"
    ssh = ["ssh", *args.ssh_opts]
    subprocess.run([*ssh, args.host, f"rm -rf {remote} && mkdir -p {args.remote_dir}"], check=True, timeout=120)
    subprocess.run(["scp", *args.ssh_opts, "-r", "-q", str(shape_dir), f"{args.host}:{args.remote_dir}/"],
                   check=True, timeout=args.timeout)
    env = f"OMP_NUM_THREADS={args.omp_threads} TRITON_BENCH_ITERS={args.iters} TRITON_BENCH_WARMUP={args.warmup}"
    script = (f"cd {remote} && for d in */; do d=${{d%/}}; echo \"== $d\"; (cd $d && chmod +x k.elf && "
              f"{env} timeout {args.per_config_timeout} ./k.elf 2>&1 | tail -4); done")
    r = subprocess.run([*ssh, args.host, script], capture_output=True, text=True, timeout=args.timeout)
    results = {}
    for block in r.stdout.split("== ")[1:]:
        name, _, body = block.partition("\n")
        t = re.search(r"Time:\s*([0-9.]+)\s*ms", body)
        c = re.search(r"Checksum:\s*(\S+)", body)
        results[name.strip()] = {"ms": float(t.group(1)) if t else None,
                                 "checksum": float(c.group(1)) if c else None, "tail": body.strip()[-160:]}
    return results


def _build_in_subprocess(info, rec, cfg, shape_dir, args, timeout, tmp):
    """Build one variant in its own process so it can run in parallel and be killed on timeout."""
    tag = "M{BLOCK_M}_N{BLOCK_N}_K{BLOCK_K}".format(**cfg)
    job = Path(tmp) / f"{info['key']}_{tag}.pkl"
    job.write_bytes(pickle.dumps((info, rec, cfg, shape_dir, args)))
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_build-one", str(job)],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return tag, cfg, None, f"compile > {timeout}s", time.time() - t0
    if r.returncode != 0:
        tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()][-1:] or [""]
        return tag, cfg, None, f"compile failed: {tail[0][:140]}", time.time() - t0
    return tag, cfg, tag, None, time.time() - t0


def tune_one(info, rec, args, workdir):
    cands = candidate_tiles(info, args)
    if args.max_configs:
        cands = cands[:args.max_configs]
    shape_dir = workdir / info["key"]
    shutil.rmtree(shape_dir, ignore_errors=True)
    built = {}
    with tempfile.TemporaryDirectory(prefix="tune-jobs-") as tmp, ThreadPoolExecutor(args.jobs) as pool:
        futs = []
        for i, cfg in enumerate(cands):
            # the baseline (x86-chosen tile) gets a longer budget so speedups can be reported
            timeout = args.baseline_timeout if i == 0 else args.compile_timeout
            futs.append(pool.submit(_build_in_subprocess, info, rec, cfg, shape_dir, args, timeout, tmp))
        for f in futs:
            tag, cfg, ok, why, secs = f.result()
            if ok:
                built[tag] = cfg
                print(f"    built {tag} in {secs:.0f}s", flush=True)
            else:
                print(f"    skip  {tag}: {why} ({secs:.0f}s)", flush=True)
    print(f"  built {len(built)}/{len(cands)} variants" + (" (compile only)" if args.compile_only else ""))
    if args.compile_only or not built:
        return None
    res = run_on_board(shape_dir, args)
    rows = []
    for name, cfg in built.items():
        r = res.get(name, {})
        rows.append({"cfg": cfg, **{k: r.get(k) for k in ("ms", "checksum", "tail")}})
    sums = [r["checksum"] for r in rows if r["checksum"] is not None]
    med = statistics.median(sums) if sums else 0.0
    for r in rows:
        r["ok"] = (r["ms"] is not None and r["checksum"] is not None
                   and abs(r["checksum"] - med) <= args.checksum_tol * (abs(med) + 1e-9))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_script", nargs="?", help="model .py to run (the only thing you change between models)")
    ap.add_argument("script_args", nargs="*", help="arguments passed through to the model script")
    ap.add_argument("--host", default=os.environ.get("TRITON_RISCV_HOST", DEFAULT_HOST))
    ap.add_argument("--remote-dir", default="~/triton-riscv-tune")
    ap.add_argument("--ssh-opts", nargs="*", default=[])
    ap.add_argument("--out", help="output dir (default artifacts/riscv/tune/<model script stem>)")
    ap.add_argument("--table", help="table json path (default <out>/gemm_table.json)")
    ap.add_argument("--only", help="comma-separated keys MxNxKxdtype to tune (default: all captured)")
    ap.add_argument("--reuse-capture", action="store_true", help="reuse <out>/capture.json instead of rerunning the model")
    ap.add_argument("--list", action="store_true", help="only list the captured GEMM kernels (and cached results)")
    ap.add_argument("--check-table", action="store_true",
                    help="capture the model afresh (honours TRITON_CPU_GEMM_TABLE) and report whether each GEMM "
                         "got its tuned tile; nothing is compiled for riscv or run on the board")
    ap.add_argument("--retune", action="store_true", help="ignore saved results and tune every shape again")
    ap.add_argument("--compile-only", action="store_true", help="build the ELFs, do not touch the board")
    ap.add_argument("--max-configs", type=int, help="cap the candidates per shape (baseline first)")
    ap.add_argument("--jobs", type=int, default=16, help="parallel variant compiles on this host")
    ap.add_argument("--compile-timeout", type=int, default=300,
                    help="skip candidate tiles whose compile takes longer than this (seconds)")
    ap.add_argument("--baseline-timeout", type=int, default=1800, help="compile budget for the baseline tile")
    ap.add_argument("--_build-one", help=argparse.SUPPRESS)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--omp-threads", type=int, default=8, help="OMP_NUM_THREADS on the board")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--per-config-timeout", type=int, default=120)
    # Pruned default grid, from the measured Qwen2.5-0.5B sweep on the BPI-F3: BLOCK_M=16 won every shape;
    # BLOCK_N was 16 (8 shapes) or 64 (2), BLOCK_K mostly 16. --full restores the wide grid.
    ap.add_argument("--full", action="store_true", help="use the wide grid instead of the pruned default")
    ap.add_argument("--block-m", type=int, nargs="+", default=None)
    ap.add_argument("--block-n", type=int, nargs="+", default=None)
    ap.add_argument("--block-k", type=int, nargs="+", default=None)
    ap.add_argument("--l1-bytes", type=int, default=65536, help="prune tiles whose A+B tiles exceed this")
    ap.add_argument("--acc-bytes", type=int, default=32768, help="prune tiles whose fp32 accumulator exceeds this")
    ap.add_argument("--max-buffer-mb", type=int, default=1024, help="skip kernels whose buffers exceed this")
    ap.add_argument("--checksum-tol", type=float, default=0.02)
    args = ap.parse_args()
    wide = args.full
    args.block_m = args.block_m or ([16, 32, 64] if wide else [16])
    args.block_n = args.block_n or ([16, 32, 64, 128, 256] if wide else [16, 64])
    args.block_k = args.block_k or ([16, 32, 64, 128] if wide else [16, 32, 64])
    if args._build_one:  # worker mode: build a single variant (used by tune_one)
        info, rec, cfg, shape_dir, wargs = pickle.loads(Path(args._build_one).read_bytes())
        build_variant(info, rec, cfg, Path(shape_dir), wargs)
        return

    if not args.model_script:
        ap.error("model_script is required")
    script = Path(args.model_script).resolve()
    out = Path(args.out or f"artifacts/riscv/tune/{script.stem}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.table) if args.table else out / "gemm_table.json"

    cap_path = out / "capture.json"
    if args.check_table:  # fresh capture that must not replace the saved (untuned-baseline) capture
        with tempfile.TemporaryDirectory(prefix="tune-inductor-") as cache:
            store = capture(script, args.script_args, cache)
    elif args.reuse_capture and cap_path.exists():
        store = json.loads(cap_path.read_text())
    else:
        with tempfile.TemporaryDirectory(prefix="tune-inductor-") as cache:
            store = capture(script, args.script_args, cache)
        cap_path.write_text(json.dumps(store))
    infos = {k: analyze(r) for k, r in store.items()}
    n_tem = sum(1 for r in store.values() if r["name"].startswith("triton_tem_"))
    print(f"[capture] {len(store)} kernels, {n_tem} GEMM templates, "
          f"{sum(r['launches'] for r in store.values())} launches")

    groups = {}
    for n, info in infos.items():
        if info is not None:
            groups.setdefault(info["key"], []).append(n)
    skipped = [k for k, r in store.items() if r["name"].startswith("triton_tem_") and infos[k] is None]
    if skipped:
        print(f"[capture] could not analyze {len(skipped)} template(s): {skipped}")
    only = set(args.only.split(",")) if args.only else None
    for key, names in groups.items():
        print(f"  {key:28s} {len(names)} kernel(s): {store[names[0]]['name']}")

    table = json.loads(table_path.read_text()) if table_path.exists() else {}
    if args.check_table:
        used = os.environ.get("TRITON_CPU_GEMM_TABLE")
        print(f"[check] TRITON_CPU_GEMM_TABLE={used or '(unset)'}; comparing with {table_path}")
        n_ok = n_bad = n_none = 0
        for key, names in groups.items():
            e = table.get(key)
            for n in names:
                tile = infos[n]["tile"]
                if not e:
                    n_none += 1
                    status = "no table entry (default tile rule)"
                elif all(tile[k] == e[k] for k in TILE_KEYS):
                    n_ok += 1
                    status = "matches table"
                else:
                    n_bad += 1
                    status = f"DIFFERS (table {e['BLOCK_M']}/{e['BLOCK_N']}/{e['BLOCK_K']})"
                print(f"  {key:22s} {store[n]['name'][:60]:60s} tile {tile['BLOCK_M']}/{tile['BLOCK_N']}/"
                      f"{tile['BLOCK_K']}  {status}")
        print(f"[check] {n_ok} match, {n_bad} differ, {n_none} without a table entry")
        return
    # A saved result is only reused if it was measured under the same conditions.
    fingerprint = {"omp_threads": args.omp_threads, "lmul": os.environ.get("TRITON_RISCV_LMUL", "8"),
                   "host": args.host}

    def cached(key):
        e = table.get(key)
        if not e or args.retune:
            return None
        fp = e.get("fingerprint")  # entries from older runs have none: treated as still valid
        return e if fp is None or fp == fingerprint else None

    todo = [k for k in groups if (not only or k in only) and cached(k) is None]
    print(f"[cache] {table_path}: {len(groups) - len(todo)}/{len(groups)} shapes already tuned "
          f"({len(todo)} to tune)")
    if args.list:
        for key in groups:
            e = cached(key)
            print(f"  {key:28s} " + (f"cached: {e['BLOCK_M']}/{e['BLOCK_N']}/{e['BLOCK_K']} {e['ms']} ms"
                                       if e else "not tuned (or stale)"))
        return
    t_start = time.time()
    for key, names in groups.items():
        if only and key not in only:
            continue
        if cached(key) is not None:
            print(f"[{key}] cached, skipping (use --retune to redo)")
            continue
        rec = store[names[0]]
        info = infos[names[0]]
        if sum(n for n, _ in rec["args"]) * 4 > args.max_buffer_mb * 2**20:
            print(f"[{key}] skipped: buffers exceed --max-buffer-mb")
            continue
        print(f"[{key}] tuning with {rec['name']} ({len(names)} kernel(s) share this shape), "
              f"baseline tile {info['tile']}")
        rows = tune_one(info, rec, args, out)
        if not rows:
            continue
        good = sorted((r for r in rows if r["ok"]), key=lambda r: r["ms"])
        for r in sorted(rows, key=lambda r: (r["ms"] is None, r["ms"] or 0))[:6]:
            flag = "ok" if r["ok"] else ("BAD-CHECKSUM" if r["ms"] is not None else "no-result")
            print(f"    {r['cfg']}  {r['ms']} ms  {flag}")
        if not good:
            print(f"[{key}] no valid result; table entry unchanged")
            continue
        base = rows[0] if rows[0]["ok"] else None
        best = good[0]
        table[key] = {**best["cfg"], "ms": best["ms"], "baseline_ms": base["ms"] if base else None,
                      "baseline_tile": info["tile"], "fingerprint": fingerprint,
                      "tuned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "candidates": [{**r["cfg"], "ms": r["ms"]} for r in good],
                      "kernels": [store[n]["name"] for n in names]}
        sp = f"{base['ms'] / best['ms']:.2f}x vs baseline" if base else "baseline invalid"
        print(f"[{key}] best {best['cfg']} {best['ms']} ms ({sp})")
        table_path.write_text(json.dumps(table, indent=2))  # save progressively
    print(f"done in {time.time() - t_start:.0f}s; table: {table_path}")
    print(f"use it: TRITON_CPU_GEMM_TABLE={table_path} python {args.model_script}")


if __name__ == "__main__":
    main()
