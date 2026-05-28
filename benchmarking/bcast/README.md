# Bcast Benchmark

This benchmark sweeps MPICH intra-node `MPI_Bcast` algorithms by changing
`MPIR_CVAR_BCAST_INTRA_ALGORITHM` through `MPI_T` during a single run.

It runs message sizes by doubling from `--min-msg-size` and appends
`--max-msg-size` when that size is not already in the sequence. With the
defaults, the sweep is:

`2, 4, 8, ..., 32768, 65532`

The measured time for each round is the slowest-rank `MPI_Bcast` duration.
The CSV stores one row per collective call, including both warmup and measured
rounds. The `phase` column labels each row as `warmup` or `actual`.

The execution order is:

`round -> message size -> algorithm`

## Build

```bash
make -C benchmarking/bcast
```

The Makefile prefers `build/install/bin/mpicxx` automatically when that in-tree
MPICH wrapper exists. To force a different wrapper:

```bash
make -C benchmarking/bcast CXX=$PWD/build/install/bin/mpicxx
```

## Run

```bash
build/install/bin/mpiexec.hydra -n 8 ./benchmarking/bcast/bcast_bench \
  --warmup-rounds 5 \
  --measured-rounds 20 \
  --output benchmarking/bcast/results.csv
```

To benchmark only a subset of algorithms:

```bash
build/install/bin/mpiexec.hydra -n 8 ./benchmarking/bcast/bcast_bench \
  --algorithms binomial,scatter_ring_allgather,tree
```

Use `--help` for the full argument list.
