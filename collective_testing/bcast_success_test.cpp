#include <mpi.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

const int kRoot = 0;
const int kMessageBytes = 4096;

std::uint8_t payload_byte(int index)
{
    return static_cast<std::uint8_t>((index * 131 + 17) & 0xff);
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

bool is_bcast_counter_line(const std::string &line)
{
    return line.find("Bcast") != std::string::npos ||
           line.find("bcast") != std::string::npos ||
           line.find("Ibcast") != std::string::npos ||
           line.find("ibcast") != std::string::npos;
}

void print_bcast_counter_confirmation(const std::string &capture_path)
{
    std::ifstream capture(capture_path.c_str());
    if (!capture) {
        std::cerr << "Could not read captured MPICH collective counter dump: "
                  << capture_path << "\n";
        return;
    }

    std::cout << "==== Bcast collective counter confirmation ====\n";
    std::string line;
    bool found = false;
    while (std::getline(capture, line)) {
        if (is_bcast_counter_line(line)) {
            std::cout << line << '\n';
            found = true;
        }
    }
    if (!found) {
        std::cout << "No Bcast collective counter entries found.\n";
    }
    std::cout << "==== END Bcast collective counter confirmation ====\n";
}

int finalize_with_bcast_counter_dump(MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    request_counter_dump(comm);

    if (rank != 0) {
        return MPI_Finalize();
    }

    char path_template[] = "/tmp/mpich_bcast_success_counters_XXXXXX";
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

    print_bcast_counter_confirmation(path_template);
    std::remove(path_template);

    return finalize_rc;
}

int expected_comm_size_from_env()
{
    const char *expected = std::getenv("BCAST_EXPECTED_RANKS");
    if (!expected || expected[0] == '\0') {
        return 0;
    }
    return std::atoi(expected);
}

} // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    MPI_Init(&argc, &argv);

    int rank = 0;
    int comm_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int expected_comm_size = expected_comm_size_from_env();
    if (expected_comm_size > 0 && comm_size != expected_comm_size) {
        if (rank == kRoot) {
            std::cerr << "ERROR: MPI_COMM_WORLD has " << comm_size
                      << " rank(s), expected " << expected_comm_size
                      << ". The MPI launcher started processes, but this MPICH "
                      << "library did not connect them into one MPI job. On "
                      << "Polaris multi-node runs, rebuild MPICH with Cray PMI "
                      << "support before launching with PALS.\n";
        }
        MPI_Finalize();
        MPI_T_finalize();
        return EXIT_FAILURE;
    }

    std::vector<std::uint8_t> buffer(kMessageBytes, 0);
    if (rank == kRoot) {
        for (int i = 0; i < kMessageBytes; i++) {
            buffer[i] = payload_byte(i);
        }
    }

    MPI_Bcast(buffer.data(), static_cast<int>(buffer.size()), MPI_BYTE, kRoot, MPI_COMM_WORLD);

    int local_success = 1;
    for (int i = 0; i < kMessageBytes; i++) {
        if (buffer[i] != payload_byte(i)) {
            local_success = 0;
            break;
        }
    }

    int global_success = 0;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == kRoot && global_success) {
        std::cout << "SUCCESS" << std::endl;
    }

    int finalize_rc = finalize_with_bcast_counter_dump(MPI_COMM_WORLD);
    MPI_T_finalize();

    if (finalize_rc != MPI_SUCCESS) {
        return finalize_rc;
    }
    return global_success ? EXIT_SUCCESS : EXIT_FAILURE;
}
