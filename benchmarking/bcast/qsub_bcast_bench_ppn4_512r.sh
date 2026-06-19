#!/bin/bash -l
#PBS -N bcast_p4_512r
#PBS -l select=128:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q prod
#PBS -A MPICH_MCS

set -euo pipefail

BENCH_DIR="/home/qktttt/mpich/benchmarking/bcast"
MPICH_PREFIX="/home/qktttt/mpich-install"
MPIEXEC="/opt/cray/pals/1.7/bin/mpiexec"

TOTAL_RANKS=512
RANKS_PER_NODE=4
WARMUP_ROUNDS=50
MEASURED_ROUNDS=50
MAX_MSG_SIZE=33554432
CPU_BIND="list:0:8:16:24"

JOB_TAG="${PBS_JOBID:-manual}"
JOB_TAG="${JOB_TAG%%.*}"
OUTPUT_CSV="${BENCH_DIR}/bcast_bench_ppn4_${TOTAL_RANKS}r_${JOB_TAG}.csv"

cd "$BENCH_DIR"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 bcast_bench.cpp -o bcast_bench

export LD_LIBRARY_PATH="$MPICH_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPIR_CVAR_BCAST_DEVICE_COLLECTIVE=0
export MPIR_CVAR_COLLECTIVE_FALLBACK=error
export MPIR_CVAR_PMI_VERSION=2
export BCAST_EXPECTED_RANKS="$TOTAL_RANKS"

echo "Running Bcast benchmark: total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} warmup=${WARMUP_ROUNDS} actual=${MEASURED_ROUNDS} max_msg_size=${MAX_MSG_SIZE} cpu_bind=${CPU_BIND}"

"$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" --cpu-bind "$CPU_BIND" ./bcast_bench \
    --warmup-rounds "$WARMUP_ROUNDS" \
    --measured-rounds "$MEASURED_ROUNDS" \
    --max-msg-size "$MAX_MSG_SIZE" \
    --output "$OUTPUT_CSV"

echo "Wrote: $OUTPUT_CSV"
