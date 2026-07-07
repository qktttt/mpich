#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

for script in qsub_bcast_bench*ppn32*r.sh; do
    echo "Submitting $script"
    qsub "$script"
done
