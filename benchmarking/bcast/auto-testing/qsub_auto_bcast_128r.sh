#!/bin/bash -l
#PBS -N auto_bcast_128r
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q debug-scaling
#PBS -A MPICH_MCS

set -e

BENCH_DIR="/home/qktttt/mpich/benchmarking/bcast/auto-testing"
MPICH_PREFIX="/home/qktttt/mpich-install"
MPIEXEC="/opt/cray/pals/1.7/bin/mpiexec"

TOTAL_RANKS=128
RANKS_PER_NODE=32
ITERATIONS=100
WARMUP_ITERATIONS=0
MIN_MSG_SIZE=32
MAX_MSG_SIZE=33554432
CPU_BIND="list:0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24:25:26:27:28:29:30:31"

JOB_TAG="${PBS_JOBID:-manual}"
JOB_TAG="${JOB_TAG%%.*}"
BUILD_DIR="${BENCH_DIR}/build-polaris"
BUILD_EXE="${BUILD_DIR}/auto_bcast_bench"
RESULT_DIR="${BENCH_DIR}/results"
ORIGINAL_RULE_JSON="${BENCH_DIR}/original_bcast_rules.json"
GENERATED_RULE_JSON="${BENCH_DIR}/generated_single_multinode_bcast_rules.json"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"
cd "$BENCH_DIR"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 \
    "${BENCH_DIR}/auto_bcast_bench.cpp" \
    -o "$BUILD_EXE"

export LD_LIBRARY_PATH="$MPICH_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPIR_CVAR_BCAST_DEVICE_COLLECTIVE=0
export MPIR_CVAR_DEVICE_COLLECTIVES=none
export MPIR_CVAR_BCAST_INTRA_ALGORITHM=auto
export MPIR_CVAR_PMI_VERSION=2
export BCAST_EXPECTED_RANKS="$TOTAL_RANKS"

run_benchmark() {
    local rule_label="$1"
    local rule_json="$2"
    local output_csv="${RESULT_DIR}/auto_bcast_${rule_label}_ppn32_${TOTAL_RANKS}r_${JOB_TAG}.csv"

    if [[ ! -f "$rule_json" ]]; then
        echo "Missing Bcast CSEL rule file: $rule_json" >&2
        exit 1
    fi

    export MPIR_CVAR_COLL_SELECTION_JSON_FILE="$rule_json"

    echo "Running auto MPI_Bcast benchmark: rule=${rule_label} total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} iterations=${ITERATIONS} min_msg_size=${MIN_MSG_SIZE} max_msg_size=${MAX_MSG_SIZE}"
    echo "Using Bcast CSEL rule file: ${MPIR_CVAR_COLL_SELECTION_JSON_FILE}"

    "$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" --cpu-bind "$CPU_BIND" \
        "$BUILD_EXE" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP_ITERATIONS" \
        --min-msg-size "$MIN_MSG_SIZE" \
        --max-msg-size "$MAX_MSG_SIZE" \
        --output "$output_csv"

    echo "Finished auto MPI_Bcast benchmark. CSV: $output_csv"
}

run_benchmark "original" "$ORIGINAL_RULE_JSON"
run_benchmark "generated_single_multinode" "$GENERATED_RULE_JSON"
