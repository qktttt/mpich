# Collective Algorithm Switch Tests

These tests switch MPICH collective algorithm CVARs with `MPI_T_cvar_write`, run
the matching MPI collective on the same logical input for every algorithm, check
correctness, and write timing/correctness rows to CSV.

The current tests cover:

```text
alltoallv_algorithm_switch_test.cpp  MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM
alltoallv_custom_algorithm_test.cpp  MPIR_CVAR_ALLTOALLV_{HIEATA,PARATA}_*
allreduce_algorithm_switch_test.cpp  MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM
bcast_algorithm_switch_test.cpp      MPIR_CVAR_BCAST_INTRA_ALGORITHM
```

Build with the MPICH you want to test:

```bash
./install/bin/mpicxx -std=c++11 collective_testing/allreduce_algorithm_switch_test.cpp \
    -o collective_testing/allreduce_algorithm_switch_test
```

If you are using the in-tree wrapper after configure:

```bash
./src/env/mpicxx -std=c++11 collective_testing/bcast_algorithm_switch_test.cpp \
    -o collective_testing/bcast_algorithm_switch_test
```

Run:

```bash
./install/bin/mpiexec -n 8 collective_testing/allreduce_algorithm_switch_test \
    --iters 5 \
    --output collective_testing/allreduce_algorithm_results.csv
```

At the end of the run, the test sets `MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS=0`
through MPI_T, captures the rank 0 MPICH collective algorithm counter dump during
`MPI_Finalize`, and reprints only the relevant collective counter entries.

Local helper scripts are split by collective:

```bash
bash collective_testing/testing.sh             # Alltoallv
bash collective_testing/testing_alltoallv_custom.sh  # Tuned hierarchical/parameterized Alltoallv
bash collective_testing/testing_allreduce.sh   # Allreduce
bash collective_testing/testing_bcast.sh       # Bcast
```

Polaris PBS scripts are also split by collective:

```bash
qsub collective_testing/testing_polaris.sh             # Alltoallv
qsub collective_testing/testing_polaris_allreduce.sh   # Allreduce
qsub collective_testing/testing_polaris_bcast.sh       # Bcast
```

## CVAR enum values

The test assumes the Alltoallv algorithm CVAR enum has this generated order:

```text
auto=0
nb=1
pairwise_sendrecv_replace=2
scattered=3
hierarchical_bruck=4
parameterized_bruck=5
```

`pairwise_sendrecv_replace` is tested with `MPI_IN_PLACE`, because that is its
MPICH restriction. The input is initialized so it represents the same logical
send data as the non-in-place algorithms.

The Allreduce test uses these generated values:

```text
auto=0
nb=1
smp=2
recursive_doubling=3
recursive_multiplying=4
reduce_scatter_allgather=5
tree=6
recexch=7
ring=8
k_reduce_scatter_allgather=9
ccl=10
release_gather=11
```

The default Allreduce test list skips `ccl` and `release_gather` because they
depend on optional backends. The baseline is `recursive_doubling`. Allreduce
counts are generated as `pof2(comm_size) * {1,2,4}` so count-restricted
algorithms can be selected on larger runs.

The Bcast test uses these generated values:

```text
auto=0
binomial=1
nb=2
circ_graph=3
smp=4
scatter_recursive_doubling_allgather=5
scatter_ring_allgather=6
pipelined_tree=7
tree=8
release_gather=9
```

The default Bcast test list skips `release_gather` because it depends on the
optional CH4 POSIX release-gather backend. The baseline is `binomial`.

The custom Alltoallv test focuses only on the two tunable custom algorithms:

```text
hierarchical_bruck   MPIR_CVAR_ALLTOALLV_HIEATA_RADIX
                     MPIR_CVAR_ALLTOALLV_HIEATA_BTHSIZE
parameterized_bruck  MPIR_CVAR_ALLTOALLV_PARATA_RADIX
```

By default it sweeps:

```text
HIEATA radix      = 2,3,4
HIEATA batch size = 1,2,4,8,32
PARATA radix      = 2,3,4
```

Run it with:

```bash
bash collective_testing/testing_alltoallv_custom.sh
```

Override the tuning lists when needed:

```bash
bash collective_testing/testing_alltoallv_custom.sh \
    --hieata-radix 2,4 \
    --hieata-batch 1,8,32 \
    --parata-radix 2,4
```

The custom test writes the requested tuning parameters to CSV and prints two
summaries at finalize: a parameter-tuple summary from the test itself
(`algorithm`, `radix`, `batch_size`) and the MPICH collective counter summary
showing the total counted `MPI_Alltoallv` calls and the per-algorithm counter
values observed by MPICH.
