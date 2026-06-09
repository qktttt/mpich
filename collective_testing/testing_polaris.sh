#!/bin/bash -l
#PBS -N alltoallv_switch
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:05:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q debug-scaling
#PBS -A MPICH_MCS

set -euo pipefail

TEST_DIR="/home/qktttt/mpich/collective_testing"
MPICH_PREFIX="/home/qktttt/mpich-install"
MPIEXEC="/opt/cray/pals/1.7/bin/mpiexec"
ITERS=5

cd "$TEST_DIR"

TEST_SRC="alltoallv_algorithm_switch_test.cpp"
TEST_BIN="./alltoallv_algorithm_switch_test"
CSV_OUT="alltoallv_algorithm_results_polaris_128r.csv"

echo "TEST_DIR=${TEST_DIR}"
echo "MPICH_PREFIX=${MPICH_PREFIX}"
echo "MPIEXEC=${MPIEXEC}"
echo "NUM_OF_NODES=4 TOTAL_NUM_RANKS=128 RANKS_PER_NODE=32"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 "$TEST_SRC" -o "$TEST_BIN"

"$MPIEXEC" --pmi=cray -n 128 --ppn 32 "$TEST_BIN" \
    --iters "$ITERS" \
    --output "$CSV_OUT"

echo "Wrote: $CSV_OUT"
