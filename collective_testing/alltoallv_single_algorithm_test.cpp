#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string label = "algorithm";
    int max_count = 32;
    bool in_place = false;
};

void usage(const char *prog)
{
    std::cerr << "Usage: " << prog << " [--label name] [--max-count N] [--in-place]\n";
}

Options parse_options(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--label" && i + 1 < argc) {
            options.label = argv[++i];
        } else if (arg == "--max-count" && i + 1 < argc) {
            options.max_count = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--in-place") {
            options.in_place = true;
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

int symmetric_count(int rank, int peer, int max_count)
{
    if (rank == peer) {
        return max_count;
    }

    int lo = std::min(rank, peer);
    int hi = std::max(rank, peer);
    int value = ((lo + 1) * 131 + (hi + 1) * 17 + max_count * 7) % (max_count + 1);

    if (((lo + hi + max_count) % 11) == 0) {
        return 0;
    }
    return value == 0 ? 1 : value;
}

int payload_value(int src, int dst, int offset)
{
    return src * 1000000 + dst * 1000 + offset;
}

void build_counts(int rank, int comm_size, int max_count,
                  std::vector<int> &counts, std::vector<int> &displs)
{
    counts.assign(comm_size, 0);
    displs.assign(comm_size, 0);

    int offset = 0;
    for (int i = 0; i < comm_size; i++) {
        counts[i] = symmetric_count(rank, i, max_count);
        displs[i] = offset;
        offset += counts[i];
    }
}

std::vector<int> build_sendbuf(int rank, const std::vector<int> &counts,
                               const std::vector<int> &displs)
{
    int total = displs.empty() ? 0 : displs.back() + counts.back();
    std::vector<int> sendbuf(total, 0);

    for (int dst = 0; dst < static_cast<int>(counts.size()); dst++) {
        for (int k = 0; k < counts[dst]; k++) {
            sendbuf[displs[dst] + k] = payload_value(rank, dst, k);
        }
    }

    return sendbuf;
}

bool check_recvbuf(int rank, const std::vector<int> &counts, const std::vector<int> &displs,
                   const std::vector<int> &recvbuf)
{
    for (int src = 0; src < static_cast<int>(counts.size()); src++) {
        for (int k = 0; k < counts[src]; k++) {
            int index = displs[src] + k;
            int expected = payload_value(src, rank, k);
            if (index >= static_cast<int>(recvbuf.size()) || recvbuf[index] != expected) {
                return false;
            }
        }
    }
    return true;
}

unsigned long long hash_buffer(const std::vector<int> &buffer)
{
    unsigned long long hash = 1469598103934665603ull;
    for (int value : buffer) {
        unsigned int word = static_cast<unsigned int>(value);
        for (int i = 0; i < 4; i++) {
            hash ^= static_cast<unsigned char>((word >> (i * 8)) & 0xffu);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    Options options = parse_options(argc, argv);

    int rank = 0;
    int comm_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    std::vector<int> counts;
    std::vector<int> displs;
    build_counts(rank, comm_size, options.max_count, counts, displs);

    std::vector<int> sendbuf = build_sendbuf(rank, counts, displs);
    std::vector<int> recvbuf;

    int mpi_errno = MPI_SUCCESS;
    if (options.in_place) {
        recvbuf = sendbuf;
        mpi_errno = MPI_Alltoallv(MPI_IN_PLACE, counts.data(), displs.data(), MPI_INT,
                                  recvbuf.data(), counts.data(), displs.data(), MPI_INT,
                                  MPI_COMM_WORLD);
    } else {
        int total_recv = displs.empty() ? 0 : displs.back() + counts.back();
        recvbuf.assign(total_recv, 0);
        mpi_errno = MPI_Alltoallv(sendbuf.data(), counts.data(), displs.data(), MPI_INT,
                                  recvbuf.data(), counts.data(), displs.data(), MPI_INT,
                                  MPI_COMM_WORLD);
    }

    int local_correct = mpi_errno == MPI_SUCCESS && check_recvbuf(rank, counts, displs, recvbuf);
    int global_correct = 0;
    int global_mpi_errno = 0;
    MPI_Allreduce(&local_correct, &global_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&mpi_errno, &global_mpi_errno, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    unsigned long long local_hash = hash_buffer(recvbuf);
    unsigned long long global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "RESULT"
                  << " label=" << options.label
                  << " ranks=" << comm_size
                  << " max_count=" << options.max_count
                  << " in_place=" << (options.in_place ? 1 : 0)
                  << " mpi_error=" << global_mpi_errno
                  << " correct=" << global_correct
                  << " hash=" << global_hash << "\n";
    }

    MPI_Finalize();
    return global_correct == 1 && global_mpi_errno == MPI_SUCCESS ? 0 : 1;
}
