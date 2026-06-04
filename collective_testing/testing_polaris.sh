#!/bin/bash -l
#PBS -N alltoallv_switch
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:15:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q debug-scaling
#PBS -A MPICH_MCS

set -euo pipefail

# Submit from the repository root with:
#   qsub -A <project> collective_testing/testing_polaris.sh
#
# Useful overrides:
#   qsub -A <project> -v MPICH_PREFIX=/path/to/mpich-install,ITERS=5 \
#       collective_testing/testing_polaris.sh

MPICH_PREFIX="${MPICH_PREFIX:-$HOME/mpich-install}"
MPICXX="${MPICXX:-$MPICH_PREFIX/bin/mpicxx}"
MPIEXEC="${MPIEXEC:-$MPICH_PREFIX/bin/mpiexec}"

if [[ -z "${PROJECT_ROOT:-}" ]]; then
    if [[ -n "${PBS_O_WORKDIR:-}" &&
          -f "${PBS_O_WORKDIR}/collective_testing/alltoallv_algorithm_switch_test.cpp" ]]; then
        PROJECT_ROOT="$PBS_O_WORKDIR"
    else
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

cd "$PROJECT_ROOT"

if [[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]]; then
    NNODES="${NNODES:-$(wc -l < "$PBS_NODEFILE")}"
else
    NNODES="${NNODES:-4}"
fi

RANKS_PER_NODE="${RANKS_PER_NODE:-32}"
NTOTRANKS="${NTOTRANKS:-$((NNODES * RANKS_PER_NODE))}"
ITERS="${ITERS:-5}"

TEST_SRC="$PROJECT_ROOT/collective_testing/alltoallv_algorithm_switch_test.cpp"
TEST_BIN="${TEST_BIN:-$PROJECT_ROOT/collective_testing/alltoallv_algorithm_switch_test}"
CSV_OUT="${CSV_OUT:-$PROJECT_ROOT/collective_testing/alltoallv_algorithm_results_polaris_${NTOTRANKS}r.csv}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MPICH_PREFIX=${MPICH_PREFIX}"
echo "NUM_OF_NODES=${NNODES} TOTAL_NUM_RANKS=${NTOTRANKS} RANKS_PER_NODE=${RANKS_PER_NODE}"

"$MPICXX" -std=c++11 -O2 "$TEST_SRC" -o "$TEST_BIN"

"$MPIEXEC" -n "$NTOTRANKS" -ppn "$RANKS_PER_NODE" "$TEST_BIN" \
    --iters "$ITERS" \
    --output "$CSV_OUT"

echo "Wrote: $CSV_OUT"
