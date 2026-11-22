# Capture the exact launch args of vLLM's unified-attention Triton kernels.
# Active only when UA_CAPTURE_DIR is set. Loaded via PYTHONPATH in every
# (spawned) python process.
import os
import sys

_DIR = os.environ.get("UA_CAPTURE_DIR")
_MOD = "vllm.v1.attention.ops.triton_unified_attention"

if _DIR:
    import importlib.abc
    import importlib.util

    def _snap(v):
        import torch
        if isinstance(v, torch.Tensor):
            return v.detach().clone().cpu()
        return v

    def _wrap(jf, name):
        orig_run = jf.run
        seen = set()

        def run(*args, grid, warmup, **kwargs):
            import torch
            q = kwargs.get("query_ptr", kwargs.get("output_ptr"))
            sl = kwargs.get("seq_lens_ptr")
            key = (name, kwargs.get("IS_3D"), tuple(q.shape), tuple(sl.tolist()))
            first = not warmup and key not in seen and len(seen) < 20
            if first:
                seen.add(key)
                assert not args, "expected keyword-only launch"
                rec = {"name": name, "grid": tuple(grid), "kwargs": {}}
                bt = kwargs.get("block_tables_ptr")
                nblk = int(bt.max().item()) + 1 if bt is not None else None
                for k, v in kwargs.items():
                    if k in ("key_cache_ptr", "value_cache_ptr"):
                        # keep the exact strided layout of blocks [0, nblk):
                        # save the raw storage span and rebuild with as_strided
                        shape = (nblk,) + tuple(v.shape[1:])
                        span = 1 + sum((n - 1) * s for n, s in zip(shape, v.stride()))
                        flat = torch.as_strided(
                            v, (span,), (1,), v.storage_offset())
                        rec["kwargs"][k] = {"storage": flat.detach().clone().cpu(),
                                            "shape": shape, "stride": tuple(v.stride())}
                        rec["kv_full_shape_" + k] = tuple(v.shape)
                    else:
                        rec["kwargs"][k] = _snap(v)
            ret = orig_run(*args, grid=grid, warmup=warmup, **kwargs)
            if first:
                torch.cuda.synchronize()
                rec["after"] = {k: kwargs[k].detach().clone().cpu()
                                for k in ("output_ptr", "segm_output_ptr",
                                          "segm_max_ptr", "segm_expsum_ptr")
                                if isinstance(kwargs.get(k), torch.Tensor)}
                mode = "" if key[1] is None else ("_3d" if key[1] else "_2d")
                tag = f"{name}{mode}_q{key[2][0]}_seqlen{'-'.join(map(str, key[3]))}"
                os.makedirs(_DIR, exist_ok=True)
                path = os.path.join(_DIR, f"{tag}.pt")
                if not os.path.exists(path):
                    torch.save(rec, path)
                    print(f"[ua-capture] saved {path} pid={os.getpid()}", file=sys.stderr)
            return ret

        jf.run = run

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            self.inner.exec_module(module)
            _wrap(module.kernel_unified_attention, "kernel_unified_attention")
            _wrap(module.reduce_segments, "reduce_segments")

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _MOD:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
