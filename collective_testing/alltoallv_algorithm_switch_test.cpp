#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Algorithm {
    const char *name;
    int cvar_value;
    bool in_place;
};

const Algorithm kAlgorithms[] = {
    {"auto", 0, false},
    {"nb", 1, false},
    {"pairwise_sendrecv_replace", 2, true},
    {"scattered", 3, false},
    {"hierarchical_bruck", 4, false},
    {"parameterized_bruck", 5, false},
};

const int kScatteredValue = 3;

//const int kDefaultSizes[] = {
//    4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
//};

const int kDefaultSizes[] = {
    4, 8, 16, 32
};

struct Options {
    std::string output = "collective_testing/alltoallv_algorithm_results.csv";
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

int symmetric_count(int rank, int peer, int max_bytes)
{
    if (rank == peer) {
        return max_bytes;
    }

    int lo = std::min(rank, peer);
    int hi = std::max(rank, peer);
    int value = ((lo + 1) * 131 + (hi + 1) * 17 + max_bytes * 7) % (max_bytes + 1);

    /* Keep a mix of zero and nonzero messages while avoiding all-zero peer pairs. */
    if (((lo + hi + max_bytes) % 11) == 0) {
        return 0;
    }
    return value == 0 ? 1 : value;
}

std::uint8_t payload_byte(int src, int dst, int offset)
{
    unsigned int value = 0x9e3779b9u;
    value ^= static_cast<unsigned int>((src + 1) * 0x45d9f3bu);
    value ^= static_cast<unsigned int>((dst + 3) * 0x119de1f3u);
    value ^= static_cast<unsigned int>((offset + 7) * 0x27d4eb2du);
    return static_cast<std::uint8_t>(value & 0xffu);
}

void build_counts(int rank, int comm_size, int max_bytes,
                  std::vector<int> &counts, std::vector<int> &displs)
{
    counts.assign(comm_size, 0);
    displs.assign(comm_size, 0);

    int offset = 0;
    for (int i = 0; i < comm_size; i++) {
        counts[i] = symmetric_count(rank, i, max_bytes);
        displs[i] = offset;
        offset += counts[i];
    }
}

std::vector<std::uint8_t> build_sendbuf(int rank, const std::vector<int> &counts,
                                        const std::vector<int> &displs)
{
    int total = displs.empty() ? 0 : displs.back() + counts.back();
    std::vector<std::uint8_t> sendbuf(total);

    for (int dst = 0; dst < static_cast<int>(counts.size()); dst++) {
        for (int k = 0; k < counts[dst]; k++) {
            sendbuf[displs[dst] + k] = payload_byte(rank, dst, k);
        }
    }

    return sendbuf;
}

std::vector<std::uint8_t> build_inplace_buffer(int rank, const std::vector<int> &counts,
                                               const std::vector<int> &displs)
{
    int total = displs.empty() ? 0 : displs.back() + counts.back();
    std::vector<std::uint8_t> recvbuf(total);

    for (int dst = 0; dst < static_cast<int>(counts.size()); dst++) {
        for (int k = 0; k < counts[dst]; k++) {
            recvbuf[displs[dst] + k] = payload_byte(rank, dst, k);
        }
    }

    return recvbuf;
}

struct RunResult {
    std::vector<std::uint8_t> recvbuf;
    double elapsed = 0.0;
    int mpi_error = MPI_SUCCESS;
};

RunResult run_alltoallv(const Algorithm &algorithm, MPI_T_cvar_handle cvar_handle,
                        const std::vector<std::uint8_t> &sendbuf,
                        const std::vector<int> &counts, const std::vector<int> &displs,
                        MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    set_cvar(cvar_handle, algorithm.cvar_value);
    MPI_Barrier(comm);

    RunResult result;
    if (algorithm.in_place) {
        result.recvbuf = build_inplace_buffer(rank, counts, displs);
        double t0 = MPI_Wtime();
        result.mpi_error = MPI_Alltoallv(MPI_IN_PLACE, counts.data(), displs.data(), MPI_BYTE,
                                         result.recvbuf.data(), counts.data(), displs.data(),
                                         MPI_BYTE, comm);
        double t1 = MPI_Wtime();
        result.elapsed = t1 - t0;
    } else {
        int total_recv = displs.empty() ? 0 : displs.back() + counts.back();
        result.recvbuf.assign(total_recv, 0);
        double t0 = MPI_Wtime();
        result.mpi_error = MPI_Alltoallv(sendbuf.data(), counts.data(), displs.data(), MPI_BYTE,
                                         result.recvbuf.data(), counts.data(), displs.data(),
                                         MPI_BYTE, comm);
        double t1 = MPI_Wtime();
        result.elapsed = t1 - t0;
    }

    MPI_Barrier(comm);
    return result;
}

bool same_buffer(const std::vector<std::uint8_t> &a, const std::vector<std::uint8_t> &b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

void write_csv_row(std::ofstream &csv, int max_message_bytes, int iteration,
                   const Algorithm &algorithm, bool is_baseline, bool correct,
                   int mpi_error, double time_min, double time_avg, double time_max,
                   long long total_bytes)
{
    csv << max_message_bytes << ','
        << iteration << ','
        << algorithm.name << ','
        << algorithm.cvar_value << ','
        << (algorithm.in_place ? 1 : 0) << ','
        << (is_baseline ? 1 : 0) << ','
        << (correct ? 1 : 0) << ','
        << mpi_error << ','
        << time_min << ','
        << time_avg << ','
        << time_max << ','
        << total_bytes << '\n';
}

void reduce_and_write_row(std::ofstream &csv, int max_message_bytes, int iteration,
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
    long long total_bytes = 0;

    MPI_Reduce(&local_correct_int, &global_correct_int, 1, MPI_INT, MPI_MIN, 0, comm);
    MPI_Reduce(&local_mpi_error, &global_mpi_error, 1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&elapsed, &time_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&elapsed, &time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&elapsed, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_bytes, &total_bytes, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    if (rank == 0) {
        write_csv_row(csv, max_message_bytes, iteration, algorithm, is_baseline,
                      global_correct_int == 1, global_mpi_error, time_min,
                      time_sum / comm_size, time_max, total_bytes);
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

    int cvar_index = find_cvar_index("MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM");
    if (cvar_index < 0) {
        if (rank == 0) {
            std::cerr << "Could not find MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM. "
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
        csv << "max_message_bytes,iteration,algorithm,cvar_value,in_place,is_baseline,"
               "correct,mpi_error,time_min_sec,time_avg_sec,time_max_sec,total_bytes\n";
    }

    const Algorithm scattered = {"scattered", kScatteredValue, false};

    for (int size_index = 0; size_index < static_cast<int>(sizeof(kDefaultSizes) / sizeof(int));
         size_index++) {
        int max_message_bytes = kDefaultSizes[size_index];
        std::vector<int> counts;
        std::vector<int> displs;
        build_counts(rank, comm_size, max_message_bytes, counts, displs);
        std::vector<std::uint8_t> sendbuf = build_sendbuf(rank, counts, displs);
        long long local_bytes = std::accumulate(counts.begin(), counts.end(), 0LL);

        std::vector<std::uint8_t> reference;
        for (int iter = 0; iter < options.iterations; iter++) {
            RunResult baseline = run_alltoallv(scattered, cvar_handle, sendbuf, counts, displs,
                                               MPI_COMM_WORLD);
            if (iter == 0) {
                reference = baseline.recvbuf;
            }
            bool local_correct = baseline.mpi_error == MPI_SUCCESS &&
                                 same_buffer(baseline.recvbuf, reference);
            reduce_and_write_row(csv, max_message_bytes, iter, scattered, true, local_correct,
                                 baseline.mpi_error, baseline.elapsed, local_bytes,
                                 MPI_COMM_WORLD);
        }

        for (const Algorithm &algorithm : kAlgorithms) {
            if (std::strcmp(algorithm.name, "scattered") == 0) {
                continue;
            }

            for (int iter = 0; iter < options.iterations; iter++) {
                RunResult result = run_alltoallv(algorithm, cvar_handle, sendbuf, counts, displs,
                                                MPI_COMM_WORLD);
                bool local_correct = result.mpi_error == MPI_SUCCESS &&
                                     same_buffer(result.recvbuf, reference);
                reduce_and_write_row(csv, max_message_bytes, iter, algorithm, false,
                                     local_correct, result.mpi_error, result.elapsed,
                                     local_bytes, MPI_COMM_WORLD);
            }
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "Wrote " << options.output << "\n";
    }

    MPI_T_cvar_handle_free(&cvar_handle);
    MPI_Finalize();
    MPI_T_finalize();
    return 0;
}
