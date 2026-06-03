# Alltoallv Algorithm Switch Test

This test switches `MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM` with MPI_T, runs
`MPI_Alltoallv` on the same logical input for every algorithm, compares each
result against `scattered`, and writes timing/correctness rows to CSV.

Build with the MPICH you want to test:

```bash
./install/bin/mpicxx -std=c++11 collective_testing/alltoallv_algorithm_switch_test.cpp \
    -o collective_testing/alltoallv_algorithm_switch_test
```

If you are using the in-tree wrapper after configure:

```bash
./src/env/mpicxx -std=c++11 collective_testing/alltoallv_algorithm_switch_test.cpp \
    -o collective_testing/alltoallv_algorithm_switch_test
```

Run:

```bash
./install/bin/mpiexec -n 8 collective_testing/alltoallv_algorithm_switch_test \
    --iters 5 \
    --output collective_testing/alltoallv_algorithm_results.csv
```

At the end of the run, the test sets `MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS=0`
through MPI_T, captures the rank 0 MPICH collective algorithm counter dump during
`MPI_Finalize`, and reprints only the `MPIR_Alltoallv_` entries.

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
