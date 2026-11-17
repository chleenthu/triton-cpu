"""Run a model script (default qwen05b_prefill_seq128.py) on x86 (native) and dump the
last launch of every Triton kernel, skipping vLLM's startup _dummy_run / profile_run, to
$QWEN_RV_DIR/<kernel>/ (default ~/qwen_rv): kernel.py, meta.pkl, in<i>.bin, and
ref<i>.bin for every buffer the launch changed. Kernels that share a name but come from
different source files (e.g. prefill vs decode graphs) get <kernel>_1, <kernel>_2, ...
A uint64 *_ptrs argument (e.g. vLLM's block_table_ptrs) holds raw addresses of other
tensors; each pointed-to tensor j of argument i is saved too, as ind<i>_<j>.bin (and
refind<i>_<j>.bin if the launch changed it), so the board run can rebuild the table.

Usage: python qwen_capture.py [script.py [script args...]]"""
import ctypes
import gc
import struct
import os
import pickle
import runpy
import shutil
import sys

os.environ["TRITON_CPU_TARGET"] = "native"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

import torch
from triton.backends.cpu.driver import CPULauncher

OUT = os.path.expanduser(os.environ.get("QWEN_RV_DIR", "~/qwen_rv"))
DIRS = {}  # (kernel name, source file) -> capture dir name


def nbytes(t):
    if t.numel() == 0:
        return 0
    return (1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))) * t.element_size()


def read(t):
    return ctypes.string_at(t.data_ptr(), nbytes(t))


def buffer_at(addr):
    """(nbytes from addr to the end of its buffer, dtype name) for the live CPU tensor
    storage or numpy array whose memory contains addr, or None. vLLM's block tables
    live in numpy-backed (UVA) buffers whose torch views may be gone by now, so numpy
    arrays are searched too; the largest containing buffer wins."""
    import numpy as np
    best = None
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor):
                if o.device.type != "cpu" or not o.numel():
                    continue
                st = o.untyped_storage()
                lo, n, dt = st.data_ptr(), st.nbytes(), str(o.dtype).removeprefix("torch.")
            elif isinstance(o, np.ndarray):
                if not o.size:
                    continue
                lo, n, dt = o.ctypes.data, o.nbytes, str(o.dtype)
            else:
                continue
        except Exception:
            continue
        if lo <= addr < lo + n and (best is None or lo + n - addr > best[0]):
            best = (lo + n - addr, dt)
    if best is None:
        # No live Python object owns the memory: fall back to the mapped region that
        # contains addr (always safe to read) and keep at most 1 MB of it. dtype is
        # unknown here; int32 matches vLLM's block tables.
        with open("/proc/self/maps") as f:
            for line in f:
                lo, hi = (int(x, 16) for x in line.split()[0].split("-"))
                if lo <= addr < hi:
                    best = (min(hi - addr, 1 << 20), "int32")
                    print(f"qwen_capture: {addr:#x} has no owning tensor; saving "
                          f"{best[0]} bytes of its mapping", flush=True)
                    break
    return best


_init = CPULauncher.__init__


def __init__(self, src, metadata):
    _init(self, src, metadata)
    self.fn = getattr(src, "fn", None)
    if self.fn is not None:
        self.sig = dict(src.signature)
        self.cst = {self.fn.arg_names[k[0]]: v for k, v in src.constants.items() if len(k) == 1}


def in_dummy_run():
    f = sys._getframe()
    while f and f.f_code.co_name not in ("_dummy_run", "profile_run"):
        f = f.f_back
    return f is not None


def __call__(self, gx, gy, gz, *rest):
    if self.fn is None or in_dummy_run():
        return self.launch(gx, gy, gz, *rest)
    fn = self.fn
    key = (fn.__name__, fn.file_name)
    if key not in DIRS:
        n = sum(k[0] == key[0] for k in DIRS)
        DIRS[key] = fn.__name__ if n == 0 else f"{fn.__name__}_{n}"
    d = os.path.join(OUT, DIRS[key])
    os.makedirs(d, exist_ok=True)
    named = list(zip(fn.arg_names, rest[6:]))
    args, before, indirect, targets = [], {}, {}, {}
    for f in os.listdir(d):
        if f.startswith(("ref", "ind")):
            os.remove(f"{d}/{f}")
    for i, (name, a) in enumerate(named):
        ty = self.sig[name]
        if isinstance(a, torch.Tensor):
            before[i] = read(a)
            with open(f"{d}/in{i}.bin", "wb") as f:
                f.write(before[i])
            args.append((name, ty, str(a.dtype).removeprefix("torch."), nbytes(a), a.data_ptr()))
            if name.endswith("_ptrs") and a.dtype == torch.uint64:
                indirect[i] = []
                for j, addr in enumerate(struct.unpack(f"{len(before[i]) // 8}Q", before[i])):
                    buf = buffer_at(addr)
                    indirect[i].append(None if buf is None else (addr, buf[0], buf[1]))
                    if buf is not None:
                        targets[i, j] = (addr, buf[0], ctypes.string_at(addr, buf[0]))
                        with open(f"{d}/ind{i}_{j}.bin", "wb") as f:
                            f.write(targets[i, j][2])
        else:
            args.append((name, ty, self.cst.get(name, a), 0, 0))
    self.launch(gx, gy, gz, *rest)
    for i, (name, a) in enumerate(named):
        if i in before and read(a) != before[i]:
            with open(f"{d}/ref{i}.bin", "wb") as f:
                f.write(read(a))
    for (i, j), (addr, n, data) in targets.items():
        after = ctypes.string_at(addr, n)
        if after != data:
            with open(f"{d}/refind{i}_{j}.bin", "wb") as f:
                f.write(after)
    shutil.copy(fn.file_name, f"{d}/kernel.py")
    with open(f"{d}/meta.pkl", "wb") as f:
        pickle.dump({"fn": fn.__name__, "grid": (gx, gy, gz), "args": args,
                     "indirect": indirect}, f)


CPULauncher.__init__ = __init__
CPULauncher.__call__ = __call__

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".py"):
        script = sys.argv.pop(1)
    else:
        script = os.path.join(here, "qwen05b_prefill_seq128.py")
    sys.argv[0] = script
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
