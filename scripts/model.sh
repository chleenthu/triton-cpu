#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/triton-cpu/.venv/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export TRITON_CACHE_DIR="$HERE/triton_cache"
rm -rf "$TRITON_CACHE_DIR"
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR="$HERE/triton_dump"
rm -rf "$TRITON_DUMP_DIR"
#export VLLM_DISABLE_COMPILE_CACHE=1
export TORCHINDUCTOR_CACHE_DIR="$HERE/inductor_cache"
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
#export VLLM_CACHE_ROOT="$TORCHINDUCTOR_CACHE_DIR"

export CC=~/llvm-project/install/bin/clang
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp
export OMP_NUM_THREADS=8
export VLLM_CPU_KVCACHE_SPACE=4

#python "$HERE/qwen05b_prefill_seq128.py" "$@"
cd ~/triton-cpu/scripts
python qwen_capture.py "$@"
python qwen_rv_run.py
