#!/bin/bash
# Tune GEMM tile sizes of a real model on the BPI-F3 board, then check the tuned tiles are picked up
# by Inductor (see scripts/tune_gemm_on_board.py).
#
#   ./tune.sh                       tune the shapes not tuned yet (saved results are reused), check the table is
#                                   picked up, then run the model with it (RUN_MODEL=0 skips the run)
#   ./tune.sh --check               only check that the model compile picks up the saved table
#   ./tune.sh --run                 only run the model script with the saved table
#   ./tune.sh --retune              tune everything again
#   ./tune.sh --list                show captured GEMM shapes and which are already tuned
#   ./tune.sh --only 1x4864x896xbf16 --full     tune one shape with the wide tile grid
#   MODEL_SCRIPT=scripts/other.py ./tune.sh     another model (only the .py name changes)
#   TABLE=/path/gemm_table.json ./tune.sh --check     explicit table
#   RUN_MODEL=0 ./tune.sh           tune and check, but do not run the model (default RUN_MODEL=1)
#
# Results: artifacts/riscv/tune/<model script name>/gemm_table.json. The model picks them up through
#   TRITON_CPU_GEMM_TABLE=<that file> python <model script>
# The check does NOT build or load any riscv code, so it works on this x86 host: it captures the
# model's GEMM kernels and prints, per kernel, the tile Inductor chose and whether it matches the table.
# Note: running the model only works where riscv64 .so files can be loaded (on the board). On an x86 host it
# fails at the end with "__triton_cpu_launcher.so: cannot open shared object file" (wrong architecture).
# Needs the patched Inductor (python scripts/patch_inductor_cpu_triton_gemm.py).
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate

RUN_MODEL=${RUN_MODEL:-1}
MODEL_SCRIPT=${MODEL_SCRIPT:-scripts/qwen_forward.py}
STEM=$(basename "$MODEL_SCRIPT" .py)
TABLE=${TABLE:-$PWD/artifacts/riscv/tune/$STEM/gemm_table.json}

# Always the project's clang (riscv64 cross-compile).
export CC=$HOME/llvm-project/install/bin/clang
export TRITON_RISCV_LMUL=${TRITON_RISCV_LMUL:-8}
# Board: user@host (also read by the tuner).
export TRITON_RISCV_HOST=${TRITON_RISCV_HOST:-chlee@140.114.78.64}

check_table() {
  [ -f "$TABLE" ] || { echo "no tuned table: $TABLE (run ./tune.sh first)"; return 1; }
  echo "using tuned GEMM table: $TABLE"
  TRITON_CPU_GEMM_TABLE=$TABLE python scripts/tune_gemm_on_board.py "$MODEL_SCRIPT" \
    --check-table --table "$TABLE" "$@"
}

run_model() {
  [ -f "$TABLE" ] || { echo "no tuned table: $TABLE (run ./tune.sh first)"; return 1; }
  echo "running $MODEL_SCRIPT with tuned GEMM table: $TABLE"
  if [ "$(uname -m)" != "riscv64" ]; then
    echo "note: host is $(uname -m); the run will fail when it loads the riscv64 launcher (expected)."
  fi
  TRITON_CPU_GEMM_TABLE=$TABLE python "$MODEL_SCRIPT" "$@"
}

if [ "$1" = "--check" ]; then
  shift
  check_table "$@"
  exit $?
fi
if [ "$1" = "--run" ]; then
  shift
  run_model "$@"
  exit $?
fi

# --reuse-capture skips re-running the model when the capture is already saved; delete
# artifacts/riscv/tune/<model script name>/capture.json to capture again (e.g. new prompt length).
python scripts/tune_gemm_on_board.py "$MODEL_SCRIPT" --reuse-capture \
  --jobs 16 --omp-threads 8 --iters 50 --warmup 5 "$@"
rc=$?
[ $rc -ne 0 ] && exit $rc

# After a real tuning run, verify Inductor picks the tuned tiles up (skipped for --list / --check-table).
case " $* " in
  *" --list "*|*" --check-table "*) ;;
  *)
    check_table || exit $?
    [ "$RUN_MODEL" = "1" ] && run_model
    ;;
esac
