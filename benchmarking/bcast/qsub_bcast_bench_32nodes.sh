#!/bin/bash -l
#PBS -N bcast_32nodes_ppns
#PBS -l select=32:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q prod
#PBS -A MPICH_MCS

set -euo pipefail

BENCH_DIR="/home/qktttt/mpich/benchmarking/bcast"
MPICH_PREFIX="/home/qktttt/mpich-install"
MPIEXEC="/opt/cray/pals/1.7/bin/mpiexec"

WARMUP_ROUNDS=50
MEASURED_ROUNDS=50
MAX_MSG_SIZE=33554432

JOB_TAG="${PBS_JOBID:-manual}"
JOB_TAG="${JOB_TAG%%.*}"

cd "$BENCH_DIR"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 bcast_bench.cpp -o bcast_bench

export LD_LIBRARY_PATH="$MPICH_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPIR_CVAR_BCAST_DEVICE_COLLECTIVE=0
export MPIR_CVAR_COLLECTIVE_FALLBACK=error
export MPIR_CVAR_PMI_VERSION=2

TOTAL_RANKS=1024
RANKS_PER_NODE=32
CPU_BIND="list:0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24:25:26:27:28:29:30:31"
OUTPUT_CSV="${BENCH_DIR}/bcast_bench_ppn32_${TOTAL_RANKS}r_${JOB_TAG}.csv"
export BCAST_EXPECTED_RANKS="$TOTAL_RANKS"
echo "Running Bcast benchmark: nodes=32 total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} cpu_bind=${CPU_BIND}"
"$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" --cpu-bind "$CPU_BIND" ./bcast_bench \
    --warmup-rounds "$WARMUP_ROUNDS" \
    --measured-rounds "$MEASURED_ROUNDS" \
    --max-msg-size "$MAX_MSG_SIZE" \
    --output "$OUTPUT_CSV"
echo "Wrote: $OUTPUT_CSV"

TOTAL_RANKS=512
RANKS_PER_NODE=16
CPU_BIND="list:0:2:4:6:8:10:12:14:16:18:20:22:24:26:28:30"
OUTPUT_CSV="${BENCH_DIR}/bcast_bench_ppn16_${TOTAL_RANKS}r_${JOB_TAG}.csv"
export BCAST_EXPECTED_RANKS="$TOTAL_RANKS"
echo "Running Bcast benchmark: nodes=32 total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} cpu_bind=${CPU_BIND}"
"$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" --cpu-bind "$CPU_BIND" ./bcast_bench \
    --warmup-rounds "$WARMUP_ROUNDS" \
    --measured-rounds "$MEASURED_ROUNDS" \
    --max-msg-size "$MAX_MSG_SIZE" \
    --output "$OUTPUT_CSV"
echo "Wrote: $OUTPUT_CSV"

TOTAL_RANKS=256
RANKS_PER_NODE=8
CPU_BIND="list:0:4:8:12:16:20:24:28"
OUTPUT_CSV="${BENCH_DIR}/bcast_bench_ppn8_${TOTAL_RANKS}r_${JOB_TAG}.csv"
export BCAST_EXPECTED_RANKS="$TOTAL_RANKS"
echo "Running Bcast benchmark: nodes=32 total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} cpu_bind=${CPU_BIND}"
"$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" --cpu-bind "$CPU_BIND" ./bcast_bench \
    --warmup-rounds "$WARMUP_ROUNDS" \
    --measured-rounds "$MEASURED_ROUNDS" \
    --max-msg-size "$MAX_MSG_SIZE" \
    --output "$OUTPUT_CSV"
echo "Wrote: $OUTPUT_CSV"

TOTAL_RANKS=128
RANKS_PER_NODE=4
CPU_BIND="list:0:8:16:24"
OUTPUT_CSV="${BENCH_DIR}/bcast_bench_ppn4_${TOTAL_RANKS}r_${JOB_TAG}.csv"
export BCAST_EXPECTED_RANKS="$TOTAL_RANKS"
echo "Running Bcast benchmark: nodes=32 total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} cpu_bind=${CPU_BIND}"
"$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" --cpu-bind "$CPU_BIND" ./bcast_bench \
    --warmup-rounds "$WARMUP_ROUNDS" \
    --measured-rounds "$MEASURED_ROUNDS" \
    --max-msg-size "$MAX_MSG_SIZE" \
    --output "$OUTPUT_CSV"
echo "Wrote: $OUTPUT_CSV"
