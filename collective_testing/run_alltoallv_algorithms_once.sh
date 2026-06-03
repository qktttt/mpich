#!/usr/bin/env bash
set -euo pipefail

MPICH_PREFIX="${MPICH_PREFIX:-$HOME/mpich-install}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/qik/projects/mpich}"
NP="${NP:-32}"
MAX_COUNT="${MAX_COUNT:-32}"

TEST_SRC="$PROJECT_ROOT/collective_testing/alltoallv_single_algorithm_test.cpp"
TEST_BIN="$PROJECT_ROOT/collective_testing/alltoallv_single_algorithm_test"

"$MPICH_PREFIX/bin/mpicxx" -std=c++11 -O2 "$TEST_SRC" -o "$TEST_BIN"

extract_field() {
    local field="$1"
    awk -v field="$field" '
        /^RESULT / {
            for (i = 1; i <= NF; i++) {
                split($i, kv, "=")
                if (kv[1] == field) {
                    print kv[2]
                    exit
                }
            }
        }
    '
}

run_case() {
    local algorithm="$1"
    local label="$2"
    local in_place_arg="${3:-}"

    MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM="$algorithm" \
        "$MPICH_PREFIX/bin/mpiexec" -n "$NP" "$TEST_BIN" \
        --label "$label" \
        --max-count "$MAX_COUNT" \
        $in_place_arg
}

echo "Building: $TEST_BIN"
echo "Ranks: $NP"
echo "Max count per peer: $MAX_COUNT"
echo

echo "Baseline: scattered"
baseline_output="$(run_case scattered baseline)"
echo "$baseline_output"
baseline_hash="$(printf '%s\n' "$baseline_output" | extract_field hash)"
baseline_correct="$(printf '%s\n' "$baseline_output" | extract_field correct)"

if [[ "$baseline_correct" != "1" ]]; then
    echo "Baseline failed correctness check." >&2
    exit 1
fi

echo
printf '%-32s %-12s %-18s %-18s %-8s %s\n' \
    "algorithm" "correct" "hash" "baseline_hash" "match" "exit"

ALGORITHMS="${ALGORITHMS:-nb pairwise_sendrecv_replace scattered hierarchical_bruck}"

for algorithm in $ALGORITHMS; do
    in_place_arg=""
    if [[ "$algorithm" == "pairwise_sendrecv_replace" ]]; then
        in_place_arg="--in-place"
    fi

    set +e
    output="$(run_case "$algorithm" "$algorithm" "$in_place_arg")"
    status=$?
    set -e

    echo "$output"

    if [[ "$status" -ne 0 ]]; then
        printf '%-32s %-12s %-18s %-18s %-8s %s\n' \
            "$algorithm" "failed" "-" "$baseline_hash" "no" "$status"
        continue
    fi

    hash="$(printf '%s\n' "$output" | extract_field hash)"
    correct="$(printf '%s\n' "$output" | extract_field correct)"
    match="no"
    if [[ "$hash" == "$baseline_hash" && "$correct" == "1" ]]; then
        match="yes"
    fi

    printf '%-32s %-12s %-18s %-18s %-8s %s\n' \
        "$algorithm" "$correct" "$hash" "$baseline_hash" "$match" "$status"
done
