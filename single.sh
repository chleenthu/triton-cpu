#!/bin/bash
# Run ONE captured Qwen kernel launch on the board, with exactly the inputs it
# sees in the full ./gen.sh run: every launch before it is replayed untimed,
# then only it is timed (see gen_qwen_driver.py --only).
#
# Usage: ./single.sh [KERNEL_NAME[@K]] [--only-phase decode] [--iters N] [--warmup N] ...
#   K = which launch of that kernel in the pass (0-based, default 0).
#   SKIP_COMPILE=1 ./single.sh ...   don't recompile even the target kernel.
# Weights are assumed already on the board (from ./gen.sh); drop --skip-weights otherwise.
source .venv/bin/activate
export CC=$HOME/llvm-project/install/bin/clang

export TRITON_EXT_ENABLED=1
export LDFLAGS="-fuse-ld=lld"
export PATH=$HOME/ccache/install/bin:$PATH

export TRITON_RISCV_LMUL=8
export TRITON_RISCV_LLVM_ARGS="-debug-only=expandpseudos"
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR=$HOME/torchinductor_cache
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp

export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=dump
export TRITON_CPU_BACKEND=1
#export TRITON_VSETVL_MINE=1

KERNEL=${1:-triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_25}
shift

PHASE=prefill
prev=
for a in "$@"; do
  [ "$prev" = --only-phase ] && PHASE=$a
  case $a in --only-phase=*) PHASE=${a#*=} ;; esac
  prev=$a
done

set -e
if [ "${SKIP_COMPILE:-0}" != 1 ]; then
  # Only the target is recompiled with the flags above; the kernels replayed
  # before it keep the .so's from the last full compile (./gen.sh).
  python scripts/compile_qwen_kernels_riscv.py --phase "$PHASE" "${KERNEL%@*}"
fi
python scripts/gen_qwen_driver.py --host chlee@140.114.78.64 --skip-weights --only "$KERNEL" "$@"
