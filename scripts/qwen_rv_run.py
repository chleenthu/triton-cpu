"""Compile every kernel captured by qwen_capture.py for riscv64, run it on the board
with the captured x86 inputs, and compare its outputs with the x86 outputs."""
import importlib.util
import os
import pickle
import subprocess
import sys

os.environ.pop("TRITON_CPU_TARGET", None)

import torch
from triton.backends.cpu.driver import ty_to_cpp
from triton.backends.cpu.riscv import build_standalone_executable, compile_kernel_to_so

OUT = os.path.expanduser(os.environ.get("QWEN_RV_DIR", "~/qwen_rv"))
HOST = os.environ.get("TRITON_RISCV_HOST", "chlee@140.114.78.64")
ITERS = 10

RUNNER = r"""#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static void *load(const char *p, size_t n) {
  void *b = aligned_alloc(64, (n + 63) / 64 * 64 + 64);
  FILE *f = fopen(p, "rb");
  if (!f || fread(b, 1, n, f) != n) { perror(p); exit(1); }
  fclose(f);
  return b;
}
static void save(const char *p, void *b, size_t n) {
  FILE *f = fopen(p, "wb");
  fwrite(b, 1, n, f);
  fclose(f);
}
extern void SYM(TYPES);
static void run(DECLS) {
  for (uint32_t x = 0; x < GX; ++x)
    for (uint32_t y = 0; y < GY; ++y)
      for (uint32_t z = 0; z < GZ; ++z)
        SYM(CALL x, y, z, GX, GY, GZ);
}
int main(void) {
LOADS
  run(NAMES);
SAVES
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < ITERS; ++i) run(NAMES);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("%.3f ms\n", ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / ITERS);
}
"""


def buffers(meta):
    """Every buffer the launch can touch: {key: (address, nbytes, dtype, input file, ref file)}.
    Pointer args are keyed by arg index i; tensors reached through a uint64 *_ptrs table
    (qwen_capture.py's "indirect") by (i, j)."""
    bufs = {i: (a[4], a[3], a[2], f"in{i}.bin", f"ref{i}.bin")
            for i, a in enumerate(meta["args"]) if a[1][0] == "*"}
    for i, entries in meta.get("indirect", {}).items():
        for j, e in enumerate(entries):
            if e is None:
                raise RuntimeError(f"{meta['args'][i][0]}[{j}]: pointed-to tensor not captured")
            bufs[i, j] = (e[0], e[1], e[2], f"ind{i}_{j}.bin", f"refind{i}_{j}.bin")
    return bufs


def groups(bufs):
    """Merge buffers whose byte ranges overlap (e.g. RoPE: out_ptr1 == out_ptr0 + 32)
    into one: {buffer key: (group, byte offset)}, [(group base, group size)]."""
    bufs = sorted((addr, addr + n, key) for key, (addr, n, *_) in bufs.items())
    where, spans = {}, []
    for lo, hi, i in bufs:
        if spans and lo < spans[-1][0] + spans[-1][1]:
            base, size = spans[-1]
            spans[-1] = (base, max(size, hi - base))
        else:
            spans.append((lo, hi - lo))
        where[i] = (len(spans) - 1, lo - spans[-1][0])
    return where, spans


def make_runner(sym, meta, where, spans, outs):
    """C main: load the groups, point every uint64 *_ptrs entry at its board copy, run."""
    types, decls, names = [], [], []
    loads = [f'  char *g{k} = load("g{k}.bin", {n});' for k, (_, n) in enumerate(spans)]
    saves = [f'  save("gout{k}.bin", g{k}, {spans[k][1]});' for k in outs]
    for i, (name, ty, val, n, _) in enumerate(meta["args"]):
        if ty == "constexpr":
            continue
        if ty[0] == "*":
            types.append("void*")
            k, off = where[i]
            loads.append(f"  void *a{i} = g{k} + {off};")
        else:
            types.append(ty_to_cpp(ty))
            loads.append(f"  {ty_to_cpp(ty)} a{i} = {val!r};")
        decls.append(f"{types[-1]} a{i}")
        names.append(f"a{i}")
    for i, entries in meta.get("indirect", {}).items():
        for j in range(len(entries)):
            k, off = where[i, j]
            loads.append(f"  ((uint64_t *)a{i})[{j}] = (uint64_t)(g{k} + {off});")
    gx, gy, gz = meta["grid"]
    return (RUNNER.replace("SYM", sym).replace("TYPES", ", ".join(types + ["uint32_t"] * 6))
            .replace("DECLS", ", ".join(decls) or "void").replace("CALL", "".join(f"{a}, " for a in names))
            .replace("NAMES", ", ".join(names)).replace("LOADS", "\n".join(loads))
            .replace("SAVES", "\n".join(saves)).replace("GX", str(gx)).replace("GY", str(gy))
            .replace("GZ", str(gz)).replace("ITERS", str(ITERS)))


def load_kernel(d, name):
    spec = importlib.util.spec_from_file_location(f"k_{name}", f"{d}/kernel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    k = getattr(mod, name)
    return k if hasattr(k, "arg_names") else k.fn


def read(data, dtype):
    return torch.frombuffer(bytearray(data), dtype=getattr(torch, dtype)).float()


def test(name):
    d = f"{OUT}/{name}"
    with open(f"{d}/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    args = meta["args"]
    sig = {a[0]: a[1] for a in args if a[1] != "constexpr"}
    cst = {a[0]: a[2] for a in args if a[1] == "constexpr"}
    compiled, so = compile_kernel_to_so(load_kernel(d, meta["fn"]), {}, constexprs=cst, signature=sig)
    sym = compiled.metadata.name

    bufs = buffers(meta)
    where, spans = groups(bufs)
    refs = [key for key in where if os.path.exists(f"{d}/{bufs[key][4]}")]
    if not refs:
        return "SKIP: launch changed no buffer"
    outs = sorted({where[i][0] for i in refs})
    blobs = [bytearray(n) for _, n in spans]
    for key, (k, off) in where.items():
        with open(f"{d}/{bufs[key][3]}", "rb") as f:
            data = f.read()
        blobs[k][off:off + len(data)] = data
    for k, blob in enumerate(blobs):
        with open(f"{d}/g{k}.bin", "wb") as f:
            f.write(blob)
    exe, lib = build_standalone_executable(sym, so, make_runner(sym, meta, where, spans, outs), f"{d}/run")

    remote = f"{os.path.basename(OUT.rstrip('/'))}/{name}"
    gs = [f"{d}/g{k}.bin" for k in range(len(spans))]
    subprocess.run(["ssh", HOST, f"rm -rf {remote} && mkdir -p {remote}"], check=True)
    subprocess.run(["scp", "-q", str(exe), str(lib), *gs, f"{HOST}:{remote}/"], check=True)
    r = subprocess.run(["ssh", HOST, f"cd {remote} && ./run"], capture_output=True, text=True)
    if r.returncode != 0:
        return f"RUN FAILED (exit {r.returncode}): {r.stderr.strip()}"
    subprocess.run(["scp", "-q", *[f"{HOST}:{remote}/gout{k}.bin" for k in outs], f"{d}/"], check=True)

    worst = 0.0
    for key in refs:
        k, off = where[key]
        _, n, dtype, _, ref_file = bufs[key]
        label = args[key][0] if isinstance(key, int) else f"{args[key[0]][0]}[{key[1]}]"
        with open(f"{d}/gout{k}.bin", "rb") as f:
            got = read(f.read()[off:off + n], dtype)
        with open(f"{d}/{ref_file}", "rb") as f:
            ref = read(f.read(), dtype)
        if not torch.allclose(got, ref, rtol=1e-2, atol=1e-2, equal_nan=True):
            return f"MISMATCH {r.stdout.strip():>12}  {label}: max err {(got - ref).abs().max().item():.4g}"
        worst = max(worst, (got - ref).abs().max().item())
    return f"PASS  {r.stdout.strip():>12}  max err {worst:.3g}"


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(os.listdir(OUT))
    for name in names:
        try:
            result = test(name)
        except Exception as e:
            result = f"ERROR {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        print(f"{name:70} {result}", flush=True)
