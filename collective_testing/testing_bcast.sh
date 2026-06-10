#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MPICH_PREFIX="${MPICH_PREFIX:-$HOME/mpich-install}"
TEST_DIR="$PROJECT_ROOT/collective_testing"
MPIEXEC="$MPICH_PREFIX/bin/mpiexec"
NP="${NP:-4}"
ITERS="${ITERS:-10}"
CSV_OUT="${CSV_OUT:-bcast_algorithm_results.csv}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$MPICH_PREFIX/lib:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$MPICH_PREFIX/lib"
fi

cd "$TEST_DIR"

TEST_SRC="bcast_algorithm_switch_test.cpp"
TEST_BIN="./bcast_algorithm_switch_test"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 "$TEST_SRC" -o "$TEST_BIN"

"$MPIEXEC" -n "$NP" "$TEST_BIN" \
    --iters "$ITERS" \
    --output "$CSV_OUT" \
    "$@"

echo "Wrote: $CSV_OUT"
