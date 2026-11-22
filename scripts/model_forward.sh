#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/triton-cpu/.venv/bin/activate

export TRITON_CACHE_DIR="$HERE/triton_cache_forward"
rm -rf "$TRITON_CACHE_DIR"
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR="$HERE/triton_dump_forward"
rm -rf "$TRITON_DUMP_DIR"
export TORCHINDUCTOR_CACHE_DIR="$HERE/inductor_cache_forward"
rm -rf "$TORCHINDUCTOR_CACHE_DIR"

export CC=~/llvm-project/install-rel/bin/clang
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp
export OMP_NUM_THREADS=8

export QWEN_RV_DIR=~/qwen_rv_forward
rm -rf "$QWEN_RV_DIR"
cd ~/triton-cpu/scripts
python qwen_capture.py qwen_forward_seq128.py "$@"
python qwen_rv_run.py
