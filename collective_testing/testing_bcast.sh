#!/usr/bin/env bash
set -euo pipefail

MPICH_PREFIX="$HOME/mpich-install"
PROJECT_ROOT="/home/qktttt/mpich"
TEST_DIR="$PROJECT_ROOT/collective_testing"
MPIEXEC="$MPICH_PREFIX/bin/mpiexec"
NP="${NP:-4}"
ITERS="${ITERS:-10}"

cd "$TEST_DIR"

TEST_SRC="bcast_algorithm_switch_test.cpp"
TEST_BIN="./bcast_algorithm_switch_test"
CSV_OUT="bcast_algorithm_results.csv"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 "$TEST_SRC" -o "$TEST_BIN"

"$MPIEXEC" -n "$NP" "$TEST_BIN" \
    --iters "$ITERS" \
    --output "$CSV_OUT"

echo "Wrote: $CSV_OUT"
