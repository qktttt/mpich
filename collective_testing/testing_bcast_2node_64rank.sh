#!/bin/bash -l
#PBS -N bcast_intra_algorithms
#PBS -l select=2:system=polaris
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
TOTAL_RANKS=64
RANKS_PER_NODE=32

BCAST_ALGORITHMS=(
    binomial
    nb
    circ_graph
    smp
    scatter_recursive_doubling_allgather
    scatter_ring_allgather
    pipelined_tree
    tree
    release_gather
)

echo "TEST_DIR=${TEST_DIR}"
echo "MPICH_PREFIX=${MPICH_PREFIX}"
echo "MPIEXEC=${MPIEXEC}"
echo "NUM_NODES=2 TOTAL_RANKS=${TOTAL_RANKS} RANKS_PER_NODE=${RANKS_PER_NODE}"

cd "$TEST_DIR"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 bcast_success_test.cpp -o bcast_success_test

export LD_LIBRARY_PATH="$MPICH_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPIR_CVAR_BCAST_DEVICE_COLLECTIVE=0
export MPIR_CVAR_COLLECTIVE_FALLBACK=error
export MPIR_CVAR_PMI_VERSION=2
export BCAST_EXPECTED_RANKS="$TOTAL_RANKS"

for algorithm in "${BCAST_ALGORITHMS[@]}"; do
    echo "===== Testing MPIR_CVAR_BCAST_INTRA_ALGORITHM=${algorithm} ====="
    export MPIR_CVAR_BCAST_INTRA_ALGORITHM="$algorithm"

    "$MPIEXEC" -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" ./bcast_success_test
done
