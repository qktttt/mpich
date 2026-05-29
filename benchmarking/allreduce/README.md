# Allreduce Benchmark

This benchmark sweeps MPICH intra-node `MPI_Allreduce` algorithms by changing
`MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM` through `MPI_T` during a single run.

It forces the benchmark through the MPIR path by setting:

- `MPIR_CVAR_DEVICE_COLLECTIVES=none`
- `MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE=0`
- `MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE=0`
- `MPIR_CVAR_COLLECTIVE_FALLBACK=error`

The benchmark intentionally excludes `release_gather`.

It runs message sizes by doubling from `--min-msg-size` and appends
`--max-msg-size` when that size is not already in the sequence. With the
defaults, the sweep is:

`2, 4, 8, ..., 32768, 65532`

The benchmark uses `MPI_UNSIGNED_CHAR` with `MPI_BXOR`, so every forced
algorithm sees a built-in, commutative reduction operator.

The measured time for each round is the slowest-rank `MPI_Allreduce` duration.
The CSV stores one row per collective call, including both warmup and measured
rounds. The `phase` column labels each row as `warmup` or `actual`.

The execution order is:

`round -> message size -> algorithm`

`reduce_scatter_allgather` in MPICH requires `count >= pof2`. Because this
benchmark measures bytes with a one-byte datatype, rows for that algorithm are
skipped when `message_size_bytes < pof2(nproc)`.

## Build

```bash
make -C benchmarking/allreduce
```

The Makefile prefers `build/install/bin/mpicxx` automatically when that in-tree
MPICH wrapper exists. To force a different wrapper:

```bash
make -C benchmarking/allreduce CXX=$PWD/build/install/bin/mpicxx
```

## Run

```bash
build/install/bin/mpiexec.hydra -n 8 ./benchmarking/allreduce/allreduce_bench \
  --warmup-rounds 5 \
  --measured-rounds 20 \
  --output benchmarking/allreduce/results.csv
```

To benchmark only a subset of algorithms:

```bash
build/install/bin/mpiexec.hydra -n 8 ./benchmarking/allreduce/allreduce_bench \
  --algorithms recursive_doubling,tree,ring
```

Use `--help` for the full argument list.
