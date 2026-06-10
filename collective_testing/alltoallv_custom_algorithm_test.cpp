#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

struct Algorithm {
    const char *name;
    int cvar_value;
};

struct TunedRun {
    Algorithm algorithm;
    int radix = 0;
    int batch_size = 0;
};

const Algorithm kHierarchicalBruck = {"hierarchical_bruck", 4};
const Algorithm kParameterizedBruck = {"parameterized_bruck", 5};

const int kDefaultSizes[] = {
    4, 8, 16
};

const int kDefaultHierarchicalRadices[] = {
    2, 3, 4
};

const int kDefaultHierarchicalBatchSizes[] = {
    1, 2, 4, 8, 32
};

const int kDefaultParameterizedRadices[] = {
    2, 3, 4
};

struct Options {
    std::string output = "alltoallv_custom_algorithm_results.csv";
    int iterations = 1;
    std::vector<int> sizes;
    std::vector<int> hier_radices;
    std::vector<int> hier_batch_sizes;
    std::vector<int> para_radices;
};

struct TunableHandles {
    MPI_T_cvar_handle algorithm = nullptr;
    MPI_T_cvar_handle hier_radix = nullptr;
    MPI_T_cvar_handle hier_batch_size = nullptr;
    MPI_T_cvar_handle para_radix = nullptr;
};

struct CvarReadback {
    int algorithm = -1;
    int hier_radix = -1;
    int hier_batch_size = -1;
    int para_radix = -1;
};

struct RunResult {
    std::vector<std::uint8_t> recvbuf;
    double elapsed = 0.0;
    int mpi_error = MPI_SUCCESS;
    CvarReadback observed;
    bool readback_match = false;
};

struct CounterEntry {
    long long count = 0;
    std::string algorithm_name;
    int cvar_value = -1;
};

struct ParameterCounterEntry {
    TunedRun run;
    long long attempted_calls = 0;
    long long successful_calls = 0;
    long long readback_match_calls = 0;
};

struct ReducedRow {
    bool root_has_data = false;
    bool global_correct = false;
    int global_mpi_error = MPI_SUCCESS;
    double time_min = 0.0;
    double time_avg = 0.0;
    double time_max = 0.0;
    long long total_bytes = 0;
    CvarReadback observed;
    bool global_readback_match = false;
};

template <size_t N>
std::vector<int> make_vector_from_array(const int (&values)[N])
{
    return std::vector<int>(values, values + N);
}

void usage(const char *prog)
{
    std::cerr << "Usage: " << prog
              << " [--output path] [--iters N] [--sizes a,b,c]"
              << " [--hieata-radix a,b,c] [--hieata-batch a,b,c]"
              << " [--parata-radix a,b,c]\n";
}

std::vector<int> parse_int_list(const char *text)
{
    std::vector<int> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        values.push_back(std::atoi(token.c_str()));
    }
    return values;
}

std::vector<int> sanitize_values(const std::vector<int> &values, int min_value)
{
    std::vector<int> sanitized;
    for (int value : values) {
        if (value < min_value) {
            continue;
        }
        if (std::find(sanitized.begin(), sanitized.end(), value) == sanitized.end()) {
            sanitized.push_back(value);
        }
    }
    return sanitized;
}

void require_non_empty(const std::vector<int> &values, const char *flag_name)
{
    if (!values.empty()) {
        return;
    }

    std::cerr << "No valid values were provided for " << flag_name << ".\n";
    std::exit(1);
}

Options parse_options(int argc, char **argv)
{
    Options options;
    options.sizes = make_vector_from_array(kDefaultSizes);
    options.hier_radices = make_vector_from_array(kDefaultHierarchicalRadices);
    options.hier_batch_sizes = make_vector_from_array(kDefaultHierarchicalBatchSizes);
    options.para_radices = make_vector_from_array(kDefaultParameterizedRadices);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            options.iterations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--sizes" && i + 1 < argc) {
            options.sizes = sanitize_values(parse_int_list(argv[++i]), 1);
            require_non_empty(options.sizes, "--sizes");
        } else if (arg == "--hieata-radix" && i + 1 < argc) {
            options.hier_radices = sanitize_values(parse_int_list(argv[++i]), 2);
            require_non_empty(options.hier_radices, "--hieata-radix");
        } else if (arg == "--hieata-batch" && i + 1 < argc) {
            options.hier_batch_sizes = sanitize_values(parse_int_list(argv[++i]), 1);
            require_non_empty(options.hier_batch_sizes, "--hieata-batch");
        } else if (arg == "--parata-radix" && i + 1 < argc) {
            options.para_radices = sanitize_values(parse_int_list(argv[++i]), 2);
            require_non_empty(options.para_radices, "--parata-radix");
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

MPI_T_cvar_handle allocate_cvar_handle_or_abort(const char *cvar_name, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    int cvar_index = find_cvar_index(cvar_name);
    if (cvar_index < 0) {
        if (rank == 0) {
            std::cerr << "Could not find " << cvar_name
                      << ". Make sure this test is linked with MPICH.\n";
        }
        MPI_Abort(comm, 1);
    }

    MPI_T_cvar_handle handle = nullptr;
    int count = 0;
    int rc = MPI_T_cvar_handle_alloc(cvar_index, nullptr, &handle, &count);
    if (rc != MPI_SUCCESS) {
        if (rank == 0) {
            std::cerr << "MPI_T_cvar_handle_alloc failed for " << cvar_name
                      << ", rc=" << rc << "\n";
        }
        MPI_Abort(comm, rc);
    }

    return handle;
}

void set_cvar(MPI_T_cvar_handle handle, int value, const char *cvar_name)
{
    int rc = MPI_T_cvar_write(handle, &value);
    if (rc != MPI_SUCCESS) {
        std::cerr << "MPI_T_cvar_write failed for " << cvar_name
                  << " with value " << value << ", rc=" << rc << "\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
}

int read_cvar(MPI_T_cvar_handle handle, const char *cvar_name)
{
    int value = -1;
    int rc = MPI_T_cvar_read(handle, &value);
    if (rc != MPI_SUCCESS) {
        std::cerr << "MPI_T_cvar_read failed for " << cvar_name
                  << ", rc=" << rc << "\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    return value;
}

CvarReadback capture_cvar_readback(const TunableHandles &handles)
{
    CvarReadback observed;
    observed.algorithm = read_cvar(handles.algorithm, "MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM");
    observed.hier_radix = read_cvar(handles.hier_radix, "MPIR_CVAR_ALLTOALLV_HIEATA_RADIX");
    observed.hier_batch_size =
        read_cvar(handles.hier_batch_size, "MPIR_CVAR_ALLTOALLV_HIEATA_BTHSIZE");
    observed.para_radix = read_cvar(handles.para_radix, "MPIR_CVAR_ALLTOALLV_PARATA_RADIX");

    return observed;
}

bool requested_cvar_readback_matches(const TunedRun &run, const CvarReadback &observed)
{
    if (observed.algorithm != run.algorithm.cvar_value) {
        return false;
    }

    if (run.algorithm.cvar_value == kHierarchicalBruck.cvar_value) {
        return observed.hier_radix == run.radix &&
               observed.hier_batch_size == run.batch_size;
    }

    if (run.algorithm.cvar_value == kParameterizedBruck.cvar_value) {
        return observed.para_radix == run.radix;
    }

    return false;
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

bool is_alltoallv_counter_line(const std::string &line)
{
    return line.find("MPIR_Alltoallv_") != std::string::npos ||
           line.find("MPIR_Ialltoallv_") != std::string::npos ||
           line.find("MPIR_TSP_Ialltoallv_") != std::string::npos;
}

int counter_name_to_cvar_value(const std::string &algorithm_name)
{
    if (algorithm_name == "MPIR_Alltoallv_intra_pairwise_sendrecv_replace") {
        return 2;
    } else if (algorithm_name == "MPIR_Alltoallv_intra_scattered") {
        return 3;
    } else if (algorithm_name == "MPIR_Alltoallv_intra_hierarchical_bruck") {
        return 4;
    } else if (algorithm_name == "MPIR_Alltoallv_intra_parameterized_bruck") {
        return 5;
    }

    return -1;
}

bool parse_counter_entry(const std::string &line, CounterEntry &entry)
{
    std::istringstream iss(line);
    if (!(iss >> entry.count >> entry.algorithm_name)) {
        return false;
    }

    entry.cvar_value = counter_name_to_cvar_value(entry.algorithm_name);
    return true;
}

void print_filtered_alltoallv_counter_dump(const std::string &capture_path,
                                           long long expected_custom_calls)
{
    std::ifstream capture(capture_path.c_str());
    if (!capture) {
        std::cerr << "Could not read captured MPICH collective counter dump: "
                  << capture_path << "\n";
        return;
    }

    std::vector<CounterEntry> entries;

    std::cout << "==== Dump Alltoallv collective algorithm counters ====\n";
    std::string line;
    while (std::getline(capture, line)) {
        if (!is_alltoallv_counter_line(line)) {
            continue;
        }

        std::cout << line << '\n';
        CounterEntry entry;
        if (parse_counter_entry(line, entry)) {
            entries.push_back(entry);
        }
    }
    std::cout << "==== END Alltoallv collective algorithm counters ====\n";

    long long counted_calls = 0;
    for (const CounterEntry &entry : entries) {
        counted_calls += entry.count;
    }

    std::cout << "==== Summary Alltoallv collective algorithm counters ====\n";
    std::cout << "Requested custom MPI_Alltoallv calls: " << expected_custom_calls << "\n";
    std::cout << "Counted Alltoallv algorithm calls: " << counted_calls << "\n";
    for (const CounterEntry &entry : entries) {
        std::cout << entry.count << "  ";
        if (entry.cvar_value >= 0) {
            std::cout << "cvar=" << entry.cvar_value;
        } else {
            std::cout << "cvar=unknown";
        }
        std::cout << "  " << entry.algorithm_name << "\n";
    }
    std::cout << "==== END Summary Alltoallv collective algorithm counters ====\n";
}

int finalize_with_alltoallv_counter_dump(MPI_Comm comm, long long expected_custom_calls)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    request_counter_dump(comm);

    if (rank != 0) {
        return MPI_Finalize();
    }

    char path_template[] = "/tmp/mpich_alltoallv_custom_counters_XXXXXX";
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

    print_filtered_alltoallv_counter_dump(path_template, expected_custom_calls);
    std::remove(path_template);

    return finalize_rc;
}

int symmetric_count(int rank, int peer, int max_bytes)
{
    if (rank == peer) {
        return max_bytes;
    }

    int lo = std::min(rank, peer);
    int hi = std::max(rank, peer);
    int value = ((lo + 1) * 131 + (hi + 1) * 17 + max_bytes * 7) % (max_bytes + 1);

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

std::vector<std::uint8_t> build_expected_recvbuf(int rank, const std::vector<int> &counts,
                                                 const std::vector<int> &displs)
{
    int total = displs.empty() ? 0 : displs.back() + counts.back();
    std::vector<std::uint8_t> recvbuf(total);

    for (int src = 0; src < static_cast<int>(counts.size()); src++) {
        for (int k = 0; k < counts[src]; k++) {
            recvbuf[displs[src] + k] = payload_byte(src, rank, k);
        }
    }

    return recvbuf;
}

RunResult run_alltoallv(const TunedRun &run, const TunableHandles &handles,
                        const std::vector<std::uint8_t> &sendbuf,
                        const std::vector<int> &counts, const std::vector<int> &displs,
                        MPI_Comm comm)
{
    MPI_Barrier(comm);
    set_cvar(handles.algorithm, run.algorithm.cvar_value, "MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM");
    if (run.algorithm.cvar_value == kHierarchicalBruck.cvar_value) {
        set_cvar(handles.hier_radix, run.radix, "MPIR_CVAR_ALLTOALLV_HIEATA_RADIX");
        set_cvar(handles.hier_batch_size, run.batch_size, "MPIR_CVAR_ALLTOALLV_HIEATA_BTHSIZE");
    } else if (run.algorithm.cvar_value == kParameterizedBruck.cvar_value) {
        set_cvar(handles.para_radix, run.radix, "MPIR_CVAR_ALLTOALLV_PARATA_RADIX");
    }
    RunResult result;
    result.observed = capture_cvar_readback(handles);
    result.readback_match = requested_cvar_readback_matches(run, result.observed);
    MPI_Barrier(comm);

    int total_recv = displs.empty() ? 0 : displs.back() + counts.back();
    result.recvbuf.assign(total_recv, 0);

    double t0 = MPI_Wtime();
    result.mpi_error = MPI_Alltoallv(sendbuf.data(), counts.data(), displs.data(), MPI_BYTE,
                                     result.recvbuf.data(), counts.data(), displs.data(),
                                     MPI_BYTE, comm);
    double t1 = MPI_Wtime();
    result.elapsed = t1 - t0;

    MPI_Barrier(comm);
    return result;
}

bool same_buffer(const std::vector<std::uint8_t> &a, const std::vector<std::uint8_t> &b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

void write_csv_row(std::ofstream &csv, int max_message_bytes, int iteration,
                   const TunedRun &run, bool correct, int mpi_error,
                   const CvarReadback &observed, bool readback_match,
                   double time_min, double time_avg, double time_max, long long total_bytes)
{
    csv << max_message_bytes << ','
        << iteration << ','
        << run.algorithm.name << ','
        << run.algorithm.cvar_value << ','
        << run.radix << ','
        << run.batch_size << ','
        << (correct ? 1 : 0) << ','
        << mpi_error << ','
        << observed.algorithm << ','
        << observed.hier_radix << ','
        << observed.hier_batch_size << ','
        << observed.para_radix << ','
        << (readback_match ? 1 : 0) << ','
        << time_min << ','
        << time_avg << ','
        << time_max << ','
        << total_bytes << '\n';
}

ReducedRow reduce_and_write_row(std::ofstream &csv, int max_message_bytes, int iteration,
                                const TunedRun &run, bool local_correct, int local_mpi_error,
                                const CvarReadback &local_observed, bool local_readback_match,
                                double elapsed, long long local_bytes, MPI_Comm comm)
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
    int global_readback_match_int = 0;
    ReducedRow summary;

    MPI_Reduce(&local_correct_int, &global_correct_int, 1, MPI_INT, MPI_MIN, 0, comm);
    MPI_Reduce(&local_mpi_error, &global_mpi_error, 1, MPI_INT, MPI_MAX, 0, comm);
    int local_readback_match_int = local_readback_match ? 1 : 0;
    MPI_Reduce(&local_readback_match_int, &global_readback_match_int, 1, MPI_INT, MPI_MIN, 0,
               comm);
    MPI_Reduce(&elapsed, &time_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&elapsed, &time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&elapsed, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_bytes, &total_bytes, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    if (rank == 0) {
        summary.root_has_data = true;
        summary.global_correct = (global_correct_int == 1);
        summary.global_mpi_error = global_mpi_error;
        summary.time_min = time_min;
        summary.time_avg = time_sum / comm_size;
        summary.time_max = time_max;
        summary.total_bytes = total_bytes;
        summary.observed = local_observed;
        summary.global_readback_match = (global_readback_match_int == 1);

        write_csv_row(csv, max_message_bytes, iteration, run, global_correct_int == 1,
                      global_mpi_error, summary.observed, summary.global_readback_match,
                      summary.time_min, summary.time_avg, summary.time_max, total_bytes);
    }

    return summary;
}

void print_parameter_counter_summary(const std::vector<ParameterCounterEntry> &entries)
{
    std::cout << "==== Summary Alltoallv parameter counters ====\n";
    long long attempted_total = 0;
    long long successful_total = 0;

    for (const ParameterCounterEntry &entry : entries) {
        attempted_total += entry.attempted_calls;
        successful_total += entry.successful_calls;
    }

    long long readback_match_total = 0;
    for (const ParameterCounterEntry &entry : entries) {
        readback_match_total += entry.readback_match_calls;
    }

    std::cout << "Attempted customized MPI_Alltoallv calls: " << attempted_total << "\n";
    std::cout << "Successful customized MPI_Alltoallv calls: " << successful_total << "\n";
    std::cout << "Readback-matched MPI_Alltoallv calls: " << readback_match_total << "\n";

    for (const ParameterCounterEntry &entry : entries) {
        std::cout << entry.attempted_calls
                  << "  successful=" << entry.successful_calls
                  << "  readback_match=" << entry.readback_match_calls
                  << "  cvar=" << entry.run.algorithm.cvar_value
                  << "  algorithm=" << entry.run.algorithm.name
                  << "  radix=" << entry.run.radix;
        if (entry.run.algorithm.cvar_value == kHierarchicalBruck.cvar_value) {
            std::cout << "  batch_size=" << entry.run.batch_size;
        }
        std::cout << "\n";
    }

    std::cout << "==== END Summary Alltoallv parameter counters ====\n";
}

std::vector<TunedRun> build_tuned_runs(const Options &options)
{
    std::vector<TunedRun> runs;

    for (int radix : options.hier_radices) {
        for (int batch_size : options.hier_batch_sizes) {
            TunedRun run;
            run.algorithm = kHierarchicalBruck;
            run.radix = radix;
            run.batch_size = batch_size;
            runs.push_back(run);
        }
    }

    for (int radix : options.para_radices) {
        TunedRun run;
        run.algorithm = kParameterizedBruck;
        run.radix = radix;
        run.batch_size = 0;
        runs.push_back(run);
    }

    return runs;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    TunableHandles handles;
    handles.algorithm =
        allocate_cvar_handle_or_abort("MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM", MPI_COMM_WORLD);
    handles.hier_radix =
        allocate_cvar_handle_or_abort("MPIR_CVAR_ALLTOALLV_HIEATA_RADIX", MPI_COMM_WORLD);
    handles.hier_batch_size =
        allocate_cvar_handle_or_abort("MPIR_CVAR_ALLTOALLV_HIEATA_BTHSIZE", MPI_COMM_WORLD);
    handles.para_radix =
        allocate_cvar_handle_or_abort("MPIR_CVAR_ALLTOALLV_PARATA_RADIX", MPI_COMM_WORLD);

    std::vector<TunedRun> tuned_runs = build_tuned_runs(options);
    long long expected_custom_calls =
        static_cast<long long>(options.sizes.size()) *
        static_cast<long long>(options.iterations) *
        static_cast<long long>(tuned_runs.size());
    std::vector<ParameterCounterEntry> parameter_counter_entries;
    parameter_counter_entries.reserve(tuned_runs.size());
    for (const TunedRun &run : tuned_runs) {
        ParameterCounterEntry entry;
        entry.run = run;
        parameter_counter_entries.push_back(entry);
    }

    std::ofstream csv;
    if (rank == 0) {
        csv.open(options.output.c_str(), std::ios::out | std::ios::trunc);
        if (!csv) {
            std::cerr << "Could not open output CSV: " << options.output << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        csv << "max_message_bytes,iteration,algorithm,cvar_value,radix,batch_size,correct,"
               "mpi_error,observed_algorithm_cvar,observed_hieata_radix,"
               "observed_hieata_batch_size,observed_parata_radix,readback_match,"
               "time_min_sec,time_avg_sec,time_max_sec,total_bytes\n";
    }

    int comm_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    for (int max_message_bytes : options.sizes) {
        std::vector<int> counts;
        std::vector<int> displs;
        build_counts(rank, comm_size, max_message_bytes, counts, displs);
        std::vector<std::uint8_t> sendbuf = build_sendbuf(rank, counts, displs);
        std::vector<std::uint8_t> expected_recvbuf =
            build_expected_recvbuf(rank, counts, displs);
        long long local_bytes = std::accumulate(counts.begin(), counts.end(), 0LL);

        for (size_t run_index = 0; run_index < tuned_runs.size(); run_index++) {
            const TunedRun &run = tuned_runs[run_index];
            for (int iter = 0; iter < options.iterations; iter++) {
                RunResult result =
                    run_alltoallv(run, handles, sendbuf, counts, displs, MPI_COMM_WORLD);
                bool local_correct = result.mpi_error == MPI_SUCCESS &&
                                     same_buffer(result.recvbuf, expected_recvbuf);
                ReducedRow summary =
                    reduce_and_write_row(csv, max_message_bytes, iter, run, local_correct,
                                         result.mpi_error, result.observed,
                                         result.readback_match, result.elapsed, local_bytes,
                                         MPI_COMM_WORLD);
                if (rank == 0) {
                    parameter_counter_entries[run_index].attempted_calls++;
                    if (summary.root_has_data && summary.global_mpi_error == MPI_SUCCESS &&
                        summary.global_correct) {
                        parameter_counter_entries[run_index].successful_calls++;
                    }
                    if (summary.root_has_data && summary.global_readback_match) {
                        parameter_counter_entries[run_index].readback_match_calls++;
                    }
                }
            }
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "Wrote " << options.output << "\n";
        print_parameter_counter_summary(parameter_counter_entries);
    }

    MPI_T_cvar_handle_free(&handles.algorithm);
    MPI_T_cvar_handle_free(&handles.hier_radix);
    MPI_T_cvar_handle_free(&handles.hier_batch_size);
    MPI_T_cvar_handle_free(&handles.para_radix);

    int finalize_rc = finalize_with_alltoallv_counter_dump(MPI_COMM_WORLD, expected_custom_calls);
    MPI_T_finalize();
    return finalize_rc == MPI_SUCCESS ? 0 : finalize_rc;
}
