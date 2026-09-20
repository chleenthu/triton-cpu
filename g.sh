#!/bin/bash

export LLVM_BUILD_DIR=$HOME/llvm-project/install
source .venv/bin/activate
# Default symbol visibility: required with shared MLIR libs, else trait TypeIDs
# mismatch between libtriton and libMLIR (ModuleOp loses its SymbolTable trait).
export TRITON_EXT_ENABLED=1
# Always build with the project's clang (never gcc). It has no default target
# triple, so give CMake's compiler check an explicit host target. Link with lld
# and compile through ccache (picked up from PATH).
export CC=$HOME/llvm-project/install/bin/clang
export CXX=$HOME/llvm-project/install/bin/clang++
export CFLAGS="--target=x86_64-unknown-linux-gnu"
export CXXFLAGS="--target=x86_64-unknown-linux-gnu"
export LDFLAGS="-fuse-ld=lld"
export PATH=$HOME/ccache/install/bin:$PATH
JSON_SYSPATH=$HOME/.triton/json TRITON_OFFLINE_BUILD=1 \
  LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
  LLVM_SYSPATH=$LLVM_BUILD_DIR \
  pip install -vvv -e .
# Faster incremental C++ rebuild (after the first full pip build):
#ninja -C build/cmake.linux-x86_64-cpython-3.10 libtriton.so

export CC=$HOME/llvm-project/install/bin/clang
source .venv/bin/activate

export TORCHINDUCTOR_CACHE_DIR=$HOME/torchinductor_cache
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp

export TRITON_BENCH_WARMUP=20
export TRITON_BENCH_ITERS=200
export OMP_NUM_THREADS=8
export TRITON_RISCV_LMUL=8

export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=dump
export TRITON_CPU_BACKEND=1
#export TRITON_VSETVL_MINE=1
#export TRITON_BRANCH_TAIL=1
#python python/tutorials/rvv_01-vector-add_elf.py
