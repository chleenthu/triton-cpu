"""Patch the installed torch/_inductor so that, with cpu_backend="triton", GEMMs
(mm/addmm/bmm) lower to Triton templates WITHOUT max_autotune / max_autotune_gemm.

Stock Inductor only considers Triton GEMM templates when a max-autotune flag is on,
and turning that flag on also makes it benchmark every candidate on the build host
(x86), which says nothing about riscv64. Six edits relax the gates for the CPU+triton
case only; every edit is marked "triton-cpu patch". Idempotent; the first run keeps
<file>.orig next to each edited file.
Tile config for GEMM templates (no benchmarking): a size-based rule over (M, N, K, dtype) picks the
candidate closest to the target tile (see GEMM_RULE); env TRITON_CPU_GEMM_CFG (e.g. "BLOCK_N=64,BLOCK_K=32")
overrides it. Revert with --revert.

Usage: python scripts/patch_inductor_cpu_triton_gemm.py [--revert]
"""
import os
import shutil
import sys

import torch

ROOT = os.path.join(os.path.dirname(torch.__file__), "_inductor")
MARK = "triton-cpu patch"

# Tile selection (no benchmarking). Precedence: TRITON_CPU_GEMM_CFG override, then the board-tuned
# table TRITON_CPU_GEMM_TABLE (from scripts/tune_gemm_on_board.py, keyed "MxNxKxdtype"), then the
# fixed default tile (TRITON_CPU_GEMM_DEFAULT, default "16,16,16") below. Env knobs: TRITON_CPU_GEMM_CFG overrides everything
# (e.g. "BLOCK_N=64,BLOCK_K=32"); TRITON_RISCV_VLEN (bits, default 256), TRITON_RISCV_LMUL
# (default 8), TRITON_CPU_L1_BYTES (default 32768) parametrize the rule.
GEMM_RULE = (
    f'        if config.cpu_backend == "triton":  # {MARK}: tile config chosen by a size-based rule, not timing\n'
    """\
            import math
            import os
            import re

            def _tile(c):
                d = str(getattr(c, "description", ""))
                return {k: int(v) for k, v in re.findall(r"\\b(BLOCK_[MNK])=(\\d+)", d)}

            want = [t.strip() for t in os.environ.get("TRITON_CPU_GEMM_CFG", "").split(",") if t.strip()]
            if want:
                for c in choices:
                    d = str(getattr(c, "description", ""))
                    if all(re.search(rf"\\b{re.escape(t)}\\b", d) for t in want):
                        return c
            else:
                from torch._inductor.virtualized import V

                tmpl = [c for c in choices if {"BLOCK_M", "BLOCK_N", "BLOCK_K"} <= _tile(c).keys()]
                if tmpl:
                    a, b = tmpl[0].input_nodes[-2:]
                    sv = V.graph.sizevars
                    hint = getattr(sv, "size_hint", None) or sv.optimization_hint  # name differs across torch versions
                    m, k = (hint(x) for x in a.get_size()[-2:])
                    n = hint(b.get_size()[-1])
                    elem = a.get_dtype().itemsize
                    dt = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
                        str(a.get_dtype()).removeprefix("torch."), str(a.get_dtype()))
                    table_path = os.environ.get("TRITON_CPU_GEMM_TABLE")
                    if table_path and os.path.exists(table_path):
                        import json

                        hit = json.load(open(table_path)).get(f"{m}x{n}x{k}x{dt}")
                        if hit:  # board-tuned tile (scripts/tune_gemm_on_board.py): nearest candidate wins
                            return min(tmpl, key=lambda c: sum(
                                abs(math.log2(_tile(c)[key] / hit[key])) for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K")))
                    def _p2(x):
                        return 1 << max(int(x) - 1, 0).bit_length()

                    # Default tile for shapes with no board-tuned entry: measured on the BPI-F3 (Qwen2.5-0.5B),
                    # 16/16/16 won 9 of 12 shapes (a 10th within 0.5%); the Inductor/x86 pick (16/128/32) was mostly the worst.
                    want_m, want_n, want_k = (int(v) for v in os.environ.get(
                        "TRITON_CPU_GEMM_DEFAULT", "16,16,16").split(","))
                    tm, tn, tk = min(_p2(m), want_m), min(_p2(n), want_n), min(_p2(k), want_k)

                    def _dist(c):
                        t = _tile(c)
                        return sum(abs(math.log2(t[key] / tgt)) for key, tgt in
                                   (("BLOCK_M", tm), ("BLOCK_N", tn), ("BLOCK_K", tk)))

                    return min(tmpl, key=_dist)
"""
)


EDITS = [
    (
        "utils.py",
        "        and (config.max_autotune or config.max_autotune_gemm or not check_max_autotune)\n",
        "        and (\n"
        "            config.max_autotune\n"
        "            or config.max_autotune_gemm\n"
        "            or not check_max_autotune\n"
        f'            or (layout.device.type == "cpu" and config.cpu_backend == "triton")  # {MARK}\n'
        "        )\n",
    ),
    (
        "utils.py",
        "def use_aten_gemm_kernels() -> bool:\n"
        "    return not (\n",
        "def use_aten_gemm_kernels() -> bool:\n"
        f'    if config.cpu_backend == "triton":  # {MARK}: no ATen candidate, Triton templates only\n'
        "        return False\n"
        "    return not (\n",
    ),
    (
        "kernel/mm.py",
        "    if (not is_nonzero) or (\n"
        "        not (inductor_config.max_autotune or inductor_config.max_autotune_gemm)\n"
        "    ):\n",
        "    if (not is_nonzero) or (\n"
        "        not (inductor_config.max_autotune or inductor_config.max_autotune_gemm)\n"
        f'        and not (layout.device.type == "cpu" and inductor_config.cpu_backend == "triton")  # {MARK}\n'
        "    ):\n",
    ),
    (
        "choices.py",
        "        # Since the following backends are not using get_mm_configs yet through the singular call,\n"
        "        if not (config.max_autotune or config.max_autotune_gemm):\n",
        f"        if (  # {MARK}: Triton templates need a fixed (non-flexible) output layout\n"
        "            len(adjusted_choices) > 0\n"
        '            and adjusted_choices[0].inputs.device_type == "cpu"\n'
        '            and config.cpu_backend == "triton"\n'
        "        ):\n"
        "            return True\n"
        "        # Since the following backends are not using get_mm_configs yet through the singular call,\n"
        "        if not (config.max_autotune or config.max_autotune_gemm):\n",
    ),
    (
        "heuristics/template/gemm.py",
        "        return inductor_config.max_autotune or inductor_config.max_autotune_gemm\n",
        "        return (\n"
        "            inductor_config.max_autotune\n"
        "            or inductor_config.max_autotune_gemm\n"
        f'            or (inputs.device_type == "cpu" and inductor_config.cpu_backend == "triton")  # {MARK}\n'
        "        )\n",
    ),
    (
        "select_algorithm.py",
        "        if config.deterministic:\n"
        "            choice = self.pick_deterministic_choice(choices)\n",
        "        if config.deterministic or (  # " + MARK + ": never benchmark on the build host\n"
        '            layout.device.type == "cpu" and config.cpu_backend == "triton"\n'
        "        ):\n"
        "            choice = self.pick_deterministic_choice(choices)\n",
    ),
    (
        "select_algorithm.py",
        '            raise AssertionError(f"expected at least 2 choices, got {len(choices)}")\n'
        "        externs = [\n",
        '            raise AssertionError(f"expected at least 2 choices, got {len(choices)}")\n'
        + GEMM_RULE
        + "        externs = [\n",
    ),
]


def main():
    revert = "--revert" in sys.argv
    for rel, old, new in EDITS:
        path = os.path.join(ROOT, rel)
        text = open(path).read()
        if revert:
            if os.path.exists(path + ".orig"):
                shutil.copy(path + ".orig", path)
                print(f"reverted {rel}")
            continue
        if new in text:
            print(f"already patched: {rel}")
            continue
        if text.count(old) != 1:
            raise SystemExit(f"{rel}: expected exactly one match for the original text, found {text.count(old)}")
        if not os.path.exists(path + ".orig"):
            shutil.copy(path, path + ".orig")
        open(path, "w").write(text.replace(old, new))
        print(f"patched {rel}")


if __name__ == "__main__":
    main()
