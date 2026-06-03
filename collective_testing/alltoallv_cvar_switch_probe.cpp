#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

struct Algorithm {
    const char *name;
    int cvar_value;
    bool in_place;
};

const Algorithm kAlgorithms[] = {
    {"nb", 1, false},
    {"pairwise_sendrecv_replace", 2, true},
    {"scattered", 3, false},
    {"hierarchical_bruck", 4, false},
    {"parameterized_bruck", 5, false},
};

int get_cvar_index(const char *name)
{
    int index = -1;
    int rc = MPI_T_cvar_get_index(name, &index);
    if (rc != MPI_SUCCESS) {
        return -1;
    }
    return index;
}

int write_cvar_and_read_back(int cvar_index, int value)
{
    MPI_T_cvar_handle handle = nullptr;
    int count = 0;

    int rc = MPI_T_cvar_handle_alloc(cvar_index, nullptr, &handle, &count);
    if (rc != MPI_SUCCESS) {
        std::cerr << "MPI_T_cvar_handle_alloc failed, rc=" << rc << "\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    rc = MPI_T_cvar_write(handle, &value);
    if (rc != MPI_SUCCESS) {
        std::cerr << "MPI_T_cvar_write failed for value " << value << ", rc=" << rc << "\n";
        MPI_T_cvar_handle_free(&handle);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    int read_back = -1;
    rc = MPI_T_cvar_read(handle, &read_back);
    if (rc != MPI_SUCCESS) {
        std::cerr << "MPI_T_cvar_read failed after writing value " << value << ", rc=" << rc
                  << "\n";
        MPI_T_cvar_handle_free(&handle);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    rc = MPI_T_cvar_handle_free(&handle);
    if (rc != MPI_SUCCESS) {
        std::cerr << "MPI_T_cvar_handle_free failed, rc=" << rc << "\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    return read_back;
}

void request_counter_dump()
{
    int cvar_index = get_cvar_index("MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS");
    if (cvar_index < 0) {
        std::cerr << "Could not find MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS\n";
        return;
    }

    int dump_rank = 0;
    write_cvar_and_read_back(cvar_index, dump_rank);
}

bool is_relevant_counter_line(const std::string &line)
{
    return line.find("MPIR_Alltoallv_") != std::string::npos ||
        line.find("MPIR_Ialltoallv_") != std::string::npos ||
        line.find("MPIR_TSP_Ialltoallv_") != std::string::npos;
}

void print_filtered_counter_dump(const std::string &capture_path)
{
    std::ifstream capture(capture_path.c_str());
    if (!capture) {
        std::cerr << "Could not read captured counter dump: " << capture_path << "\n";
        return;
    }

    std::cout << "==== Relevant Alltoallv collective algorithm counters ====\n";
    std::string line;
    while (std::getline(capture, line)) {
        if (is_relevant_counter_line(line)) {
            std::cout << line << "\n";
        }
    }
    std::cout << "==== END relevant Alltoallv collective algorithm counters ====\n";
}

int finalize_with_filtered_counter_dump(MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    request_counter_dump();

    if (rank != 0) {
        return MPI_Finalize();
    }

    char path_template[] = "/tmp/mpich_alltoallv_switch_probe_XXXXXX";
    int capture_fd = mkstemp(path_template);
    if (capture_fd < 0) {
        std::cerr << "Could not create temporary file for counter dump capture.\n";
        return MPI_Finalize();
    }

    std::cout.flush();
    std::fflush(stdout);

    int saved_stdout = dup(STDOUT_FILENO);
    if (saved_stdout < 0 || dup2(capture_fd, STDOUT_FILENO) < 0) {
        std::cerr << "Could not redirect stdout for counter dump capture.\n";
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

    print_filtered_counter_dump(path_template);
    std::remove(path_template);
    return finalize_rc;
}

bool check_recvbuf(const std::vector<int> &recvbuf, int rank, int comm_size)
{
    for (int src = 0; src < comm_size; src++) {
        int expected = src * 1000 + rank;
        if (recvbuf[src] != expected) {
            return false;
        }
    }
    return true;
}

bool run_one_algorithm(const Algorithm &algorithm, int alltoallv_cvar_index, MPI_Comm comm)
{
    int rank = 0;
    int comm_size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    int read_back = write_cvar_and_read_back(alltoallv_cvar_index, algorithm.cvar_value);
    if (rank == 0) {
        std::cout << "request name=" << algorithm.name << " cvar_value=" << algorithm.cvar_value
                  << " read_back=" << read_back << " in_place=" << (algorithm.in_place ? 1 : 0)
                  << "\n";
    }

    std::vector<int> counts(comm_size, 1);
    std::vector<int> displs(comm_size, 0);
    for (int i = 0; i < comm_size; i++) {
        displs[i] = i;
    }

    std::vector<int> sendbuf(comm_size, 0);
    std::vector<int> recvbuf(comm_size, 0);
    for (int dst = 0; dst < comm_size; dst++) {
        sendbuf[dst] = rank * 1000 + dst;
    }

    MPI_Barrier(comm);
    int mpi_errno = MPI_SUCCESS;
    if (algorithm.in_place) {
        recvbuf = sendbuf;
        mpi_errno = MPI_Alltoallv(MPI_IN_PLACE, counts.data(), displs.data(), MPI_INT,
                                  recvbuf.data(), counts.data(), displs.data(), MPI_INT, comm);
    } else {
        mpi_errno = MPI_Alltoallv(sendbuf.data(), counts.data(), displs.data(), MPI_INT,
                                  recvbuf.data(), counts.data(), displs.data(), MPI_INT, comm);
    }
    MPI_Barrier(comm);

    bool local_ok = mpi_errno == MPI_SUCCESS && check_recvbuf(recvbuf, rank, comm_size);
    int local_ok_int = local_ok ? 1 : 0;
    int global_ok_int = 0;
    MPI_Reduce(&local_ok_int, &global_ok_int, 1, MPI_INT, MPI_MIN, 0, comm);

    if (rank == 0) {
        std::cout << "result name=" << algorithm.name << " mpi_errno=" << mpi_errno
                  << " correct=" << global_ok_int << "\n";
    }

    return local_ok;
}

} // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int alltoallv_cvar_index = get_cvar_index("MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM");
    if (alltoallv_cvar_index < 0) {
        if (rank == 0) {
            std::cerr << "Could not find MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_all_ok = 1;
    for (const Algorithm &algorithm : kAlgorithms) {
        bool local_ok = run_one_algorithm(algorithm, alltoallv_cvar_index, MPI_COMM_WORLD);
        if (!local_ok) {
            local_all_ok = 0;
        }
    }

    int global_all_ok = 0;
    MPI_Allreduce(&local_all_ok, &global_all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "all_requested_runs_correct=" << global_all_ok << "\n";
    }

    int finalize_rc = finalize_with_filtered_counter_dump(MPI_COMM_WORLD);
    MPI_T_finalize();
    return finalize_rc == MPI_SUCCESS && global_all_ok == 1 ? 0 : 1;
}
