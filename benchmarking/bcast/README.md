# Bcast Benchmark

This benchmark measures `MPI_Bcast` for the algorithm selected by the launch
environment. It no longer changes `MPIR_CVAR_BCAST_INTRA_ALGORITHM` internally;
the PBS job scripts do that by launching the benchmark once per algorithm and
appending all results to one CSV.

The job-script sweep uses the same algorithms as
`collective_testing/testing_bcast_2node_64rank.sh`:

`binomial, nb, circ_graph, smp, scatter_recursive_doubling_allgather, scatter_ring_allgather, pipelined_tree, tree, release_gather`

For output filenames, the job scripts shorten `pipelined_tree` to `pipet` so
those CSVs do not match resume checks for the `tree` algorithm.

Message sizes include each doubling level and the halfway point before the next
level. With the defaults, the sweep is:

`2, 3, 4, 6, 8, 12, ..., 8388608, 12582912, 16777216, 25165824, 33554432`

That is, the default run now covers up to 32 MiB.

Each `MPI_Bcast` is preceded by `MPI_Barrier`. Every rank measures its local
`MPI_Bcast` duration, then rank 0 receives the sum with `MPI_Reduce` and
divides by `nproc` to store `avg_latency_sec`. The CSV also includes min and
max rank times so average-vs-slowest-rank latency can be compared later.

Rank 0 stores one row per collective call in memory and writes the CSV at the
end of the program. Warmup and measured iterations are both written; the
`phase` column labels each row as `warmup` or `actual`. Rank 0 also writes a
separate collective-count CSV named like the timing output with
`_collective_counts` before the extension. It records how many `MPI_Bcast`
calls were made for the configured algorithm label.

The execution order is:

`message size -> iteration`

## Build

```bash
/home/qktttt/mpich-install/bin/mpicxx -std=c++11 -O2 \
  benchmarking/bcast/bcast_bench.cpp \
  -o benchmarking/bcast/bcast_bench
```

Or replace the wrapper path with another MPICH `mpicxx`:

```bash
mpicxx -std=c++11 -O2 benchmarking/bcast/bcast_bench.cpp \
  -o benchmarking/bcast/bcast_bench
```

## Run

```bash
export BCAST_EXPECTED_RANKS=8
export MPIR_CVAR_BCAST_DEVICE_COLLECTIVE=0
export MPIR_CVAR_DEVICE_COLLECTIVES=none
export MPIR_CVAR_COLLECTIVE_FALLBACK=error
export MPIR_CVAR_BCAST_INTRA_ALGORITHM=binomial
export BCAST_ALGORITHM_LABEL=binomial
mpiexec -n 8 ./benchmarking/bcast/bcast_bench \
  --warmup-rounds 5 \
  --measured-rounds 20 \
  --output benchmarking/bcast/results.csv
```

`BCAST_EXPECTED_RANKS` is optional, but useful on systems where a launcher can
start processes that do not connect into one `MPI_COMM_WORLD`; the benchmark
fails early if the communicator size does not match.

To append another algorithm run into the same CSVs:

```bash
export MPIR_CVAR_BCAST_INTRA_ALGORITHM=tree
export BCAST_ALGORITHM_LABEL=tree
mpiexec -n 8 ./benchmarking/bcast/bcast_bench \
  --warmup-rounds 5 \
  --measured-rounds 20 \
  --output benchmarking/bcast/results.csv \
  --append-output
```

Use `--help` for the full argument list.

## Polaris ppn sweep jobs

The node-count PBS wrappers run all ppn variants for one allocation. For
example, the 1-node wrapper runs ppn 32, 16, 8, and 4 in sequence before the job
exits:

```bash
qsub benchmarking/bcast/qsub_bcast_bench_1node.sh
qsub benchmarking/bcast/qsub_bcast_bench_2nodes.sh
qsub benchmarking/bcast/qsub_bcast_bench_4nodes.sh
qsub benchmarking/bcast/qsub_bcast_bench_8nodes.sh
qsub benchmarking/bcast/qsub_bcast_bench_16nodes.sh
qsub benchmarking/bcast/qsub_bcast_bench_32nodes.sh
```

Each wrapper is self-contained. It compiles once, sources
`run_bcast_algorithm_sweep.sh`, and then launches every configured algorithm for
each ppn run. The helper sets `MPIR_CVAR_BCAST_INTRA_ALGORITHM` and
`BCAST_ALGORITHM_LABEL` before each launch, appending timing rows to the main
CSV and count rows to the matching `_collective_counts.csv`.

The node-count wrappers run four explicit ppn configurations:

`ppn32 -> list:0:1:2:...:31`

`ppn16 -> list:0:2:4:...:30`

`ppn8 -> list:0:4:8:...:28`

`ppn4 -> list:0:8:16:24`

Output files still include the effective total rank count. A 16-node sweep
writes `bcast_bench_ppn32_512r_<job>.csv`,
`bcast_bench_ppn16_256r_<job>.csv`,
`bcast_bench_ppn8_128r_<job>.csv`, and
`bcast_bench_ppn4_64r_<job>.csv`.

The run order is `ppn32`, then `ppn16`, then `ppn8`, then `ppn4`.
