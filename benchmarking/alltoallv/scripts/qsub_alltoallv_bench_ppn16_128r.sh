#!/bin/bash -l
#PBS -N alltoallv_p16_128r
#PBS -l select=8:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q preemptable
#PBS -A MPICH_MCS

BENCH_DIR="/home/qktttt/mpich/benchmarking/alltoallv"
MPICH_PREFIX="/home/qktttt/mpich-install"
MPIEXEC="/opt/cray/pals/1.7/bin/mpiexec"

TOTAL_RANKS=128
RANKS_PER_NODE=16
WARMUP_ROUNDS=20
MEASURED_ROUNDS=20
MAX_MSG_SIZE=33554432
CPU_BIND="list:0:2:4:6:8:10:12:14:16:18:20:22:24:26:28:30"
ALLTOALLV_ALGORITHMS="nb,scattered,hierarchical_bruck,hierarchical_shm_alltoallv"
JOB_TAG="${PBS_JOBID:-manual}"
JOB_TAG="${JOB_TAG%%.*}"
OUTPUT_CSV="${BENCH_DIR}/alltoallv_bench_ppn${RANKS_PER_NODE}_${TOTAL_RANKS}r_${JOB_TAG}.csv"

cd "$BENCH_DIR"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 alltoallv_bench.cpp -o alltoallv_bench

benchmark_already_done() {
    local output_prefix="${OUTPUT_CSV%.csv}"
    local candidate

    output_prefix="${output_prefix%_${JOB_TAG}}"

    for candidate in "${output_prefix}"_*.csv; do
        [ -e "$candidate" ] || continue

        if [ "$(wc -l < "$candidate")" -gt 1 ]; then
            echo "Skipping Alltoallv benchmark: total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE}; existing result: ${candidate}"
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
export MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE=0
export MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE=0
export MPIR_CVAR_COLLECTIVE_FALLBACK=error
export MPIR_CVAR_PMI_VERSION=2
export ALLTOALLV_EXPECTED_RANKS="$TOTAL_RANKS"

echo "Running Alltoallv benchmark: algorithms=${ALLTOALLV_ALGORITHMS} total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} warmup=${WARMUP_ROUNDS} actual=${MEASURED_ROUNDS} max_msg_size=${MAX_MSG_SIZE} cpu_bind=${CPU_BIND}"

rm -f "$OUTPUT_CSV"

"$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" --cpu-bind "$CPU_BIND" ./alltoallv_bench \
        --warmup-rounds "$WARMUP_ROUNDS" \
        --measured-rounds "$MEASURED_ROUNDS" \
        --max-msg-size "$MAX_MSG_SIZE" \
        --output "$OUTPUT_CSV" \
        --algorithms "${ALLTOALLV_ALGORITHMS}"

echo "Wrote: $OUTPUT_CSV"
echo "Finished Alltoallv benchmark sweep."
