#!/bin/bash
# triton-cpu: replay vLLM unified-attention kernels and dump IR (ttir, ttcir, ...)
#   ./run.sh                 # default cases (real 128-token prefill + 3D)
#   ./run.sh --case all      # every captured call
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/triton-cpu/.venv/bin/activate
cd "$HERE"

export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR="$HERE/triton_dump"
export TRITON_CACHE_DIR="$HERE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$HERE/inductor_cache"
rm -rf "$TRITON_DUMP_DIR" "$TRITON_CACHE_DIR"

export CC=$HOME/llvm-project/install-rel/bin/clang
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp

export TRITON_CPU_TARGET=native
export OMP_NUM_THREADS=8
export TRITON_BENCH_WARMUP=0
export TRITON_BENCH_ITERS=1
export TRITON_RISCV_LMUL=8
export TRITON_LLVM_DEBUG_ONLY="analyze-tail-masks"

export TRITON_CPU_BACKEND=1
export TRITON_VSETVL_CHUNK=1
python unified_attention_repro.py "$@"
