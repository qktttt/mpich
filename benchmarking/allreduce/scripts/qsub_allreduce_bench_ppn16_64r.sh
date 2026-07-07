#!/bin/bash -l
#PBS -N allreduce_p16_64r
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:30:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q preemptable
#PBS -A MPICH_MCS

BENCH_DIR="/home/qktttt/mpich/benchmarking/allreduce"
MPICH_PREFIX="/home/qktttt/mpich-install"
MPIEXEC="/opt/cray/pals/1.7/bin/mpiexec"

TOTAL_RANKS=64
RANKS_PER_NODE=16
WARMUP_ROUNDS=250
MEASURED_ROUNDS=250
MAX_MSG_SIZE=33554432
CPU_BIND="list:0:2:4:6:8:10:12:14:16:18:20:22:24:26:28:30"
ALLREDUCE_ALGORITHMS="nb,smp,recursive_doubling,recursive_multiplying,reduce_scatter_allgather,tree,recexch,ring,k_reduce_scatter_allgather,hierarchical"

JOB_TAG="${PBS_JOBID:-manual}"
JOB_TAG="${JOB_TAG%%.*}"
OUTPUT_CSV="${BENCH_DIR}/allreduce_bench_ppn${RANKS_PER_NODE}_${TOTAL_RANKS}r_${JOB_TAG}.csv"

cd "$BENCH_DIR"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 allreduce_bench.cpp -o allreduce_bench

benchmark_already_done() {
    local output_prefix="${OUTPUT_CSV%.csv}"
    local candidate

    output_prefix="${output_prefix%_${JOB_TAG}}"

    for candidate in "${output_prefix}"_*.csv; do
        [ -e "$candidate" ] || continue

        if [ "$(wc -l < "$candidate")" -gt 1 ]; then
            echo "Skipping Allreduce benchmark: total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE}; existing result: ${candidate}"
            return 0
        fi
    done

    return 1
}

if benchmark_already_done; then
    exit 0
fi

export LD_LIBRARY_PATH="$MPICH_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPIR_CVAR_DEVICE_COLLECTIVES=none
export MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE=0
export MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE=0
export MPIR_CVAR_COLLECTIVE_FALLBACK=error
export MPIR_CVAR_PMI_VERSION=2
export ALLREDUCE_EXPECTED_RANKS="$TOTAL_RANKS"

echo "Running Allreduce benchmark: algorithms=${ALLREDUCE_ALGORITHMS} total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} warmup=${WARMUP_ROUNDS} actual=${MEASURED_ROUNDS} max_msg_size=${MAX_MSG_SIZE} cpu_bind=${CPU_BIND}"

rm -f "$OUTPUT_CSV"

"$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" --cpu-bind "$CPU_BIND" ./allreduce_bench \
        --warmup-rounds "$WARMUP_ROUNDS" \
        --measured-rounds "$MEASURED_ROUNDS" \
        --max-msg-size "$MAX_MSG_SIZE" \
        --output "$OUTPUT_CSV" \
        --algorithms "$ALLREDUCE_ALGORITHMS"

echo "Wrote: $OUTPUT_CSV"
echo "Finished Allreduce benchmark sweep."
