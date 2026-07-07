#!/bin/bash -l
#PBS -N bcast_p4_16r
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:30:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q preemptable
#PBS -A MPICH_MCS

BENCH_DIR="/home/qktttt/mpich/benchmarking/bcast"
MPICH_PREFIX="/home/qktttt/mpich-install"
MPIEXEC="/opt/cray/pals/1.7/bin/mpiexec"

TOTAL_RANKS=16
RANKS_PER_NODE=4
WARMUP_ROUNDS=250
MEASURED_ROUNDS=250
MAX_MSG_SIZE=33554432
CPU_BIND="list:0:8:16:24"

JOB_TAG="${PBS_JOBID:-manual}"
JOB_TAG="${JOB_TAG%%.*}"
OUTPUT_CSV="${BENCH_DIR}/bcast_bench_ppn4_${TOTAL_RANKS}r_${JOB_TAG}.csv"

cd "$BENCH_DIR"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 bcast_bench.cpp -o bcast_bench

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

algorithm_output_label() {
    local algorithm="$1"

    case "$algorithm" in
        pipelined_tree)
            echo "pipet"
            ;;
        *)
            echo "$algorithm"
            ;;
    esac
}

benchmark_already_done() {
    local algorithm="$1"
    local output_label="$2"
    local output_prefix="${OUTPUT_CSV%.csv}"
    local candidate

    output_prefix="${output_prefix%_${JOB_TAG}}"

    for candidate in "${output_prefix}"_*_"${output_label}".csv; do
        [ -e "$candidate" ] || continue

        if [ "$(wc -l < "$candidate")" -gt 1 ]; then
            echo "Skipping Bcast benchmark: algorithm=${algorithm} total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE}; existing result: ${candidate}"
            return 0
        fi
    done

    return 1
}

export LD_LIBRARY_PATH="$MPICH_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPIR_CVAR_BCAST_DEVICE_COLLECTIVE=0
export MPIR_CVAR_DEVICE_COLLECTIVES=none
export MPIR_CVAR_COLLECTIVE_FALLBACK=error
export MPIR_CVAR_PMI_VERSION=2
export BCAST_EXPECTED_RANKS="$TOTAL_RANKS"

for ALGORITHM in "${BCAST_ALGORITHMS[@]}"; do
    ALGORITHM_OUTPUT_LABEL="$(algorithm_output_label "$ALGORITHM")"
    ALGORITHM_OUTPUT_CSV="${OUTPUT_CSV%.csv}_${ALGORITHM_OUTPUT_LABEL}.csv"
    ALGORITHM_COUNTER_CSV="${ALGORITHM_OUTPUT_CSV%.csv}_collective_counts.csv"

    if benchmark_already_done "$ALGORITHM" "$ALGORITHM_OUTPUT_LABEL"; then
        continue
    fi

    export MPIR_CVAR_BCAST_INTRA_ALGORITHM="$ALGORITHM"
    export BCAST_ALGORITHM_LABEL="$ALGORITHM"

    echo "Running Bcast benchmark: algorithm=${ALGORITHM} total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} warmup=${WARMUP_ROUNDS} actual=${MEASURED_ROUNDS} max_msg_size=${MAX_MSG_SIZE} cpu_bind=${CPU_BIND}"

    rm -f "$ALGORITHM_OUTPUT_CSV" "$ALGORITHM_COUNTER_CSV"

    "$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" --cpu-bind "$CPU_BIND" ./bcast_bench \
        --warmup-rounds "$WARMUP_ROUNDS" \
        --measured-rounds "$MEASURED_ROUNDS" \
        --max-msg-size "$MAX_MSG_SIZE" \
        --output "$ALGORITHM_OUTPUT_CSV" \
        --counter-output "$ALGORITHM_COUNTER_CSV" \
        --algorithm-label "$ALGORITHM"

    echo "Wrote: $ALGORITHM_OUTPUT_CSV"
    echo "Wrote: $ALGORITHM_COUNTER_CSV"
done

echo "Wrote one timing CSV and one counter CSV per algorithm under: ${OUTPUT_CSV%.csv}_<algorithm-output-label>.csv"
