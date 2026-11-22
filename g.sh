#!/bin/bash
cd ~/triton-cpu
source .venv/bin/activate
export LLVM_BUILD_DIR=$HOME/llvm-project/install-rel
#export TRITON_EXT_ENABLED=1
#export CC=$HOME/llvm-project/install-rel/bin/clang
#export CXX=$HOME/llvm-project/install-rel/bin/clang++
#export CFLAGS="--target=x86_64-unknown-linux-gnu"
#export CXXFLAGS="--target=x86_64-unknown-linux-gnu"
#export LDFLAGS="-fuse-ld=lld"
#export PATH=$HOME/ccache/install/bin:$PATH
#export TRITON_SLEEF_SOURCE_DIR=$PWD/third_party/sleef
#JSON_SYSPATH=$HOME/.triton/json TRITON_OFFLINE_BUILD=1 TRITON_BUILD_NVIDIA_GSAN_RUNTIME=0 \
#  LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
#  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
#  LLVM_SYSPATH=$LLVM_BUILD_DIR \
#  pip install --ignore-requires-python -vvv -e .
ninja -C build/cmake.linux-x86_64-cpython-3.10 libtriton.so

source .venv/bin/activate
export CC=$HOME/llvm-project/install-rel/bin/clang
export TORCHINDUCTOR_CACHE_DIR=$HOME/torchinductor_cache
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp

#export TRITON_BENCH_WARMUP=20
#export TRITON_BENCH_ITERS=200
export OMP_NUM_THREADS=8
export TRITON_RISCV_LMUL=8
#export TRITON_RISCV_LLVM_ARGS="-debug-only=expandpseudos"
#export TRITON_LLVM_DEBUG_ONLY="analyze-tail-masks,tail-mask-to-evl,convert-memory-ops"
#export TRITON_LLVM_DEBUG_ONLY="convert-memory-ops,expandpseudos"

cd python/tutorials/cpu
KERNEL=${KERNEL:-mask_branch}
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=artifacts/dump/$KERNEL
export TRITON_CPU_BACKEND=1
#export TRITON_CPU_FLIP_TO_SHUFFLE=1
export TRITON_VSETVL_CHUNK=1
python rvv_"$KERNEL"_elf.py

export TRITON_DUMP_DIR=artifacts/dump/$KERNEL
export TRITON_VSETVL_REDUCE=1
#python python/tutorials/cpu/rvv_03-matrix-multiplication_elf.py
