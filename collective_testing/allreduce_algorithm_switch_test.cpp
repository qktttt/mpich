#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

struct Algorithm {
    const char *name;
    int cvar_value;
};

const Algorithm kAlgorithms[] = {
    {"nb", 1},
    {"smp", 2},
    {"recursive_doubling", 3},
    {"recursive_multiplying", 4},
    {"reduce_scatter_allgather", 5},
    {"tree", 6},
    {"recexch", 7},
    {"ring", 8},
    {"k_reduce_scatter_allgather", 9},
};

const int kBaselineValue = 3;

const int kDefaultCountMultipliers[] = {
    1, 2, 4
};

struct Options {
    std::string output = "collective_testing/allreduce_algorithm_results.csv";
    int iterations = 1;
};

void usage(const char *prog)
{
    std::cerr << "Usage: " << prog << " [--output path] [--iters N]\n";
}

Options parse_options(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            options.iterations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    return options;
}

int find_cvar_index(const char *target_name)
{
    int num_cvars = 0;
    if (MPI_T_cvar_get_num(&num_cvars) != MPI_SUCCESS) {
        return -1;
    }

    for (int i = 0; i < num_cvars; i++) {
        char name[MPI_MAX_OBJECT_NAME] = {0};
        int name_len = MPI_MAX_OBJECT_NAME;
        int verbosity = 0;
        MPI_Datatype datatype = MPI_DATATYPE_NULL;
        MPI_T_enum enumtype = MPI_T_ENUM_NULL;
        char desc[1] = {0};
        int desc_len = 1;
        int bind = 0;
        int scope = 0;

        int rc = MPI_T_cvar_get_info(i, name, &name_len, &verbosity, &datatype, &enumtype,
                                     desc, &desc_len, &bind, &scope);
        if (rc == MPI_SUCCESS && std::strcmp(name, target_name) == 0) {
            return i;
        }
    }

    return -1;
}

void set_cvar(MPI_T_cvar_handle handle, int value)
{
    int rc = MPI_T_cvar_write(handle, &value);
    if (rc != MPI_SUCCESS) {
        std::cerr << "MPI_T_cvar_write failed for value " << value << ", rc=" << rc << "\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
}

void request_counter_dump(MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    int cvar_index = find_cvar_index("MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS");
    if (cvar_index < 0) {
        if (rank == 0) {
            std::cerr << "Could not find MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS; "
                      << "MPICH collective algorithm counters will not be dumped.\n";
        }
        return;
    }

    MPI_T_cvar_handle handle = nullptr;
    int count = 0;
    int rc = MPI_T_cvar_handle_alloc(cvar_index, nullptr, &handle, &count);
    if (rc != MPI_SUCCESS) {
        if (rank == 0) {
            std::cerr << "MPI_T_cvar_handle_alloc failed for "
                      << "MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS, rc=" << rc << "\n";
        }
        return;
    }

    int dump_rank = 0;
    rc = MPI_T_cvar_write(handle, &dump_rank);
    if (rc != MPI_SUCCESS && rank == 0) {
        std::cerr << "MPI_T_cvar_write failed for MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS, "
                  << "rc=" << rc << "\n";
    }

    MPI_T_cvar_handle_free(&handle);
}

void print_filtered_allreduce_counter_dump(const std::string &capture_path)
{
    std::ifstream capture(capture_path.c_str());
    if (!capture) {
        std::cerr << "Could not read captured MPICH collective counter dump: "
                  << capture_path << "\n";
        return;
    }

    std::cout << "==== Dump Allreduce collective algorithm counters ====\n";
    std::string line;
    while (std::getline(capture, line)) {
        if (line.find("MPIR_Allreduce_") != std::string::npos ||
            line.find("MPIR_Iallreduce_") != std::string::npos ||
            line.find("MPIR_TSP_Iallreduce_") != std::string::npos ||
            line.find("MPIR_Coll_nb") != std::string::npos) {
            std::cout << line << '\n';
        }
    }
    std::cout << "==== END Allreduce collective algorithm counters ====\n";
}

int finalize_with_allreduce_counter_dump(MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    request_counter_dump(comm);

    if (rank != 0) {
        return MPI_Finalize();
    }

    char path_template[] = "/tmp/mpich_allreduce_counters_XXXXXX";
    int capture_fd = mkstemp(path_template);
    if (capture_fd < 0) {
        std::cerr << "Could not create temporary file for MPICH counter dump capture.\n";
        return MPI_Finalize();
    }

    std::cout.flush();
    std::fflush(stdout);

    int saved_stdout = dup(STDOUT_FILENO);
    if (saved_stdout < 0 || dup2(capture_fd, STDOUT_FILENO) < 0) {
        std::cerr << "Could not redirect stdout for MPICH counter dump capture.\n";
        close(capture_fd);
        if (saved_stdout >= 0) {
            close(saved_stdout);
        }
        std::remove(path_template);
        return MPI_Finalize();
    }
    close(capture_fd);

    int finalize_rc = MPI_Finalize();

    std::fflush(stdout);
    dup2(saved_stdout, STDOUT_FILENO);
    close(saved_stdout);

    print_filtered_allreduce_counter_dump(path_template);
    std::remove(path_template);

    return finalize_rc;
}

int nearest_power_of_two_leq(int value)
{
    int pof2 = 1;
    while (pof2 * 2 <= value) {
        pof2 *= 2;
    }
    return pof2;
}

std::vector<int> build_sendbuf(int rank, int count)
{
    std::vector<int> sendbuf(count);
    for (int i = 0; i < count; i++) {
        sendbuf[i] = rank * 1000 + i * 7 + 3;
    }
    return sendbuf;
}

std::vector<int> build_expected(int comm_size, int count)
{
    std::vector<int> expected(count);
    int rank_sum = comm_size * (comm_size - 1) / 2;
    for (int i = 0; i < count; i++) {
        expected[i] = rank_sum * 1000 + comm_size * (i * 7 + 3);
    }
    return expected;
}

struct RunResult {
    std::vector<int> recvbuf;
    double elapsed = 0.0;
    int mpi_error = MPI_SUCCESS;
};

RunResult run_allreduce(const Algorithm &algorithm, MPI_T_cvar_handle cvar_handle,
                        const std::vector<int> &sendbuf, MPI_Comm comm)
{
    MPI_Barrier(comm);
    set_cvar(cvar_handle, algorithm.cvar_value);
    MPI_Barrier(comm);

    RunResult result;
    result.recvbuf.assign(sendbuf.size(), 0);
    double t0 = MPI_Wtime();
    result.mpi_error = MPI_Allreduce(sendbuf.data(), result.recvbuf.data(),
                                     static_cast<int>(sendbuf.size()), MPI_INT, MPI_SUM, comm);
    double t1 = MPI_Wtime();
    result.elapsed = t1 - t0;

    MPI_Barrier(comm);
    return result;
}

bool same_buffer(const std::vector<int> &a, const std::vector<int> &b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

void write_csv_row(std::ofstream &csv, int count, int iteration, const Algorithm &algorithm,
                   bool is_baseline, bool correct, int mpi_error, double time_min,
                   double time_avg, double time_max, long long total_input_bytes)
{
    csv << count << ','
        << iteration << ','
        << algorithm.name << ','
        << algorithm.cvar_value << ','
        << (is_baseline ? 1 : 0) << ','
        << (correct ? 1 : 0) << ','
        << mpi_error << ','
        << time_min << ','
        << time_avg << ','
        << time_max << ','
        << total_input_bytes << '\n';
}

void reduce_and_write_row(std::ofstream &csv, int count, int iteration,
                          const Algorithm &algorithm, bool is_baseline, bool local_correct,
                          int local_mpi_error, double elapsed, long long local_bytes,
                          MPI_Comm comm)
{
    int rank = 0;
    int comm_size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    int local_correct_int = local_correct ? 1 : 0;
    int global_correct_int = 0;
    int global_mpi_error = 0;
    double time_min = 0.0;
    double time_sum = 0.0;
    double time_max = 0.0;
    long long total_input_bytes = 0;

    MPI_Reduce(&local_correct_int, &global_correct_int, 1, MPI_INT, MPI_MIN, 0, comm);
    MPI_Reduce(&local_mpi_error, &global_mpi_error, 1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&elapsed, &time_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&elapsed, &time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&elapsed, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_bytes, &total_input_bytes, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    if (rank == 0) {
        write_csv_row(csv, count, iteration, algorithm, is_baseline,
                      global_correct_int == 1, global_mpi_error, time_min,
                      time_sum / comm_size, time_max, total_input_bytes);
    }
}

} // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    Options options = parse_options(argc, argv);

    int rank = 0;
    int comm_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int cvar_index = find_cvar_index("MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM");
    if (cvar_index < 0) {
        if (rank == 0) {
            std::cerr << "Could not find MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM. "
                      << "Make sure this test is linked with MPICH.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_T_cvar_handle cvar_handle = nullptr;
    int cvar_count = 0;
    int rc = MPI_T_cvar_handle_alloc(cvar_index, nullptr, &cvar_handle, &cvar_count);
    if (rc != MPI_SUCCESS) {
        if (rank == 0) {
            std::cerr << "MPI_T_cvar_handle_alloc failed, rc=" << rc << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    std::ofstream csv;
    if (rank == 0) {
        csv.open(options.output.c_str(), std::ios::out | std::ios::trunc);
        if (!csv) {
            std::cerr << "Could not open output CSV: " << options.output << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        csv << "count,iteration,algorithm,cvar_value,is_baseline,correct,mpi_error,"
               "time_min_sec,time_avg_sec,time_max_sec,total_input_bytes\n";
    }

    const Algorithm baseline_algorithm = {"recursive_doubling", kBaselineValue};
    int pof2 = nearest_power_of_two_leq(comm_size);

    for (int size_index = 0;
         size_index < static_cast<int>(sizeof(kDefaultCountMultipliers) / sizeof(int));
         size_index++) {
        int count = pof2 * kDefaultCountMultipliers[size_index];
        std::vector<int> sendbuf = build_sendbuf(rank, count);
        std::vector<int> expected = build_expected(comm_size, count);
        long long local_bytes = static_cast<long long>(count) * sizeof(int);

        for (int iter = 0; iter < options.iterations; iter++) {
            RunResult baseline = run_allreduce(baseline_algorithm, cvar_handle, sendbuf,
                                               MPI_COMM_WORLD);
            bool local_correct = baseline.mpi_error == MPI_SUCCESS &&
                                 same_buffer(baseline.recvbuf, expected);
            reduce_and_write_row(csv, count, iter, baseline_algorithm, true, local_correct,
                                 baseline.mpi_error, baseline.elapsed, local_bytes,
                                 MPI_COMM_WORLD);
        }

        for (const Algorithm &algorithm : kAlgorithms) {
            if (std::strcmp(algorithm.name, "recursive_doubling") == 0) {
                continue;
            }

            for (int iter = 0; iter < options.iterations; iter++) {
                RunResult result = run_allreduce(algorithm, cvar_handle, sendbuf,
                                                 MPI_COMM_WORLD);
                bool local_correct = result.mpi_error == MPI_SUCCESS &&
                                     same_buffer(result.recvbuf, expected);
                reduce_and_write_row(csv, count, iter, algorithm, false, local_correct,
                                     result.mpi_error, result.elapsed, local_bytes,
                                     MPI_COMM_WORLD);
            }
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "Wrote " << options.output << "\n";
    }

    MPI_T_cvar_handle_free(&cvar_handle);
    int finalize_rc = finalize_with_allreduce_counter_dump(MPI_COMM_WORLD);
    MPI_T_finalize();
    return finalize_rc == MPI_SUCCESS ? 0 : finalize_rc;
}
