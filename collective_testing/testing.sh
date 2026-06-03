#!/usr/bin/env bash
set -euo pipefail

MPICH_PREFIX="$HOME/mpich-install"


TEST_SRC="/collective_testing/alltoallv_algorithm_switch_test.cpp"
TEST_BIN="/collective_testing/alltoallv_algorithm_switch_test"
CSV_OUT="/collective_testing/alltoallv_algorithm_results.csv"

PROJECT_ROOT="/home/qik/projects/mpich"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 $PROJECT_ROOT/$TEST_SRC -o $PROJECT_ROOT/$TEST_BIN

"$MPICH_PREFIX/bin/mpiexec" -n 32 $PROJECT_ROOT/$TEST_BIN \
  --iters 10 \
  --output $PROJECT_ROOT/$CSV_OUT

echo "Wrote: $CSV_OUT"