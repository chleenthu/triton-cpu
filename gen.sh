#!/bin/bash
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
export TRITON_DUMP_DIR=/home/chlee/triton-cpu/dump_sta
export TRITON_CPU_BACKEND=1
python scripts/compile_qwen_kernels_riscv.py
python scripts/gen_qwen_driver.py --host chlee@140.114.78.64
