"""Run qwen05b_prefill_seq128.py on x86 (native) and dump the last launch of every
Triton kernel, skipping vLLM's startup _dummy_run / profile_run, to
~/qwen_rv/<kernel>/: kernel.py, meta.pkl, in<i>.bin, and ref<i>.bin for every
buffer the launch changed."""
import ctypes
import os
import pickle
import shutil
import sys

os.environ["TRITON_CPU_TARGET"] = "native"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

import torch
from triton.backends.cpu.driver import CPULauncher

OUT = os.path.expanduser("~/qwen_rv")


def nbytes(t):
    if t.numel() == 0:
        return 0
    return (1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))) * t.element_size()


def read(t):
    return ctypes.string_at(t.data_ptr(), nbytes(t))


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
    fn, d = self.fn, os.path.join(OUT, self.fn.__name__)
    os.makedirs(d, exist_ok=True)
    named = list(zip(fn.arg_names, rest[6:]))
    args, before = [], {}
    for f in os.listdir(d):
        if f.startswith("ref"):
            os.remove(f"{d}/{f}")
    for i, (name, a) in enumerate(named):
        ty = self.sig[name]
        if isinstance(a, torch.Tensor):
            before[i] = read(a)
            with open(f"{d}/in{i}.bin", "wb") as f:
                f.write(before[i])
            args.append((name, ty, str(a.dtype).removeprefix("torch."), nbytes(a), a.data_ptr()))
        else:
            args.append((name, ty, self.cst.get(name, a), 0, 0))
    self.launch(gx, gy, gz, *rest)
    for i, (name, a) in enumerate(named):
        if i in before and read(a) != before[i]:
            with open(f"{d}/ref{i}.bin", "wb") as f:
                f.write(read(a))
    shutil.copy(fn.fn.__code__.co_filename, f"{d}/kernel.py")
    with open(f"{d}/meta.pkl", "wb") as f:
        pickle.dump({"fn": fn.__name__, "grid": (gx, gy, gz), "args": args}, f)


CPULauncher.__init__ = __init__
CPULauncher.__call__ = __call__

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qwen05b_prefill_seq128
    qwen05b_prefill_seq128.main()
