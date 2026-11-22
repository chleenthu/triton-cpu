#!/bin/bash
# triton-cpu: riscv64 flow -- cross-compile the captured vLLM unified-attention
# launches to standalone ELFs (rvv_*_elf.py style), run them on the board and
# check against vLLM's outputs. Dumps IR (ttir, ttcir, ..., asm) per kernel.
#   ./run_riscv.sh                   # default cases
#   ./run_riscv.sh --case all        # every captured call
#   ./run_riscv.sh --compile-only    # build only, don't touch the board
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/triton-cpu/.venv/bin/activate
cd "$HERE"

export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR="$HERE/triton_dump_riscv"
export TRITON_CACHE_DIR="$HERE/triton_cache_riscv"
export TORCHINDUCTOR_CACHE_DIR="$HERE/inductor_cache_riscv"
rm -rf "$TRITON_DUMP_DIR" "$TRITON_CACHE_DIR"

export CC=$HOME/llvm-project/install-rel/bin/clang
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp

export OMP_NUM_THREADS=8
export TRITON_BENCH_WARMUP=0
export TRITON_BENCH_ITERS=1
export TRITON_RISCV_LMUL=${TRITON_RISCV_LMUL:-1}
export TRITON_LLVM_DEBUG_ONLY="analyze-tail-masks"
#export TRITON_RISCV_LLVM_ARGS="-debug-only=expandpseudos"

export TRITON_CPU_BACKEND=1
export TRITON_VSETVL_CHUNK=1
python rvv_unified_attention_elf.py "$@"
