#!/bin/bash

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

run_bcast_algorithm_sweep() {
    local cpu_bind_args=()

    export MPIR_CVAR_COLLECTIVE_FALLBACK=error

    if [[ -n "${CPU_BIND:-}" ]]; then
        cpu_bind_args=(--cpu-bind "$CPU_BIND")
    fi

    for algorithm in "${BCAST_ALGORITHMS[@]}"; do
        local algorithm_output_label
        local algorithm_output_csv
        local algorithm_counter_csv

        algorithm_output_label="$(algorithm_output_label "$algorithm")"
        algorithm_output_csv="${OUTPUT_CSV%.csv}_${algorithm_output_label}.csv"
        algorithm_counter_csv="${algorithm_output_csv%.csv}_collective_counts.csv"

        if benchmark_already_done "$algorithm" "$algorithm_output_label"; then
            continue
        fi

        export MPIR_CVAR_BCAST_INTRA_ALGORITHM="$algorithm"
        export BCAST_ALGORITHM_LABEL="$algorithm"

        echo "Running Bcast benchmark: algorithm=${algorithm} total_ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE} warmup=${WARMUP_ROUNDS} actual=${MEASURED_ROUNDS} max_msg_size=${MAX_MSG_SIZE} cpu_bind=${CPU_BIND:-none}"

        rm -f "$algorithm_output_csv" "$algorithm_counter_csv"

        "$MPIEXEC" --pmi=cray -n "$TOTAL_RANKS" --ppn "$RANKS_PER_NODE" "${cpu_bind_args[@]}" ./bcast_bench \
            --warmup-rounds "$WARMUP_ROUNDS" \
            --measured-rounds "$MEASURED_ROUNDS" \
            --max-msg-size "$MAX_MSG_SIZE" \
            --output "$algorithm_output_csv" \
            --counter-output "$algorithm_counter_csv" \
            --algorithm-label "$algorithm"

        echo "Wrote: $algorithm_output_csv"
        echo "Wrote: $algorithm_counter_csv"
    done
}
