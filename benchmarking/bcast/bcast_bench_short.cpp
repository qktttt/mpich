#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

/* Rank 0 is both the broadcast root and the rank that owns CSV output. */
constexpr int kRoot = 0;
constexpr int kDefaultMaxMsgSize = 32 * 1024 * 1024;
const int kDefaultMessageSizes[] = {
    2,       3,       4,       6,     8,   12,  32768, 12582912
};

/* Command-line configuration and benchmark defaults. */
struct Config {
    int warmup_rounds = 5;
    int measured_rounds = 20;
    int min_msg_size = 2;
    int max_msg_size = kDefaultMaxMsgSize;
    std::string output_path = "benchmarking/bcast/bcast_bench.csv";
    std::string counter_output_path;
    std::string algorithm_label;
    bool append_output = false;
    bool show_help = false;
};

/* One CSV row. Rank 0 stores these in memory and writes them after the sweep. */
struct ResultRow {
    std::string phase;
    int phase_iteration = 0;
    int global_iteration = 0;
    std::string algorithm;
    int root = kRoot;
    int nproc = 0;
    int message_size_bytes = 0;
    double time_sum_sec = 0.0;
    double avg_latency_sec = 0.0;
    double min_time_sec = 0.0;
    double max_time_sec = 0.0;
    bool correct = false;
};

struct CounterDumpRow {
    std::string requested_collective;
    std::string requested_algorithm;
    std::string internal_collective;
    std::string internal_algorithm;
    int nproc = 0;
    int dump_rank = 0;
    long long count = 0;
    std::string raw_line;
};

bool file_exists(const std::string &path);

std::string csv_field(const std::string &value)
{
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

std::string infer_internal_collective(const std::string &algorithm)
{
    const std::string mpir_tsp_prefix = "MPIR_TSP_";
    if (algorithm.find(mpir_tsp_prefix) == 0) {
        const std::size_t start = mpir_tsp_prefix.size();
        const std::size_t end = algorithm.find('_', start);
        if (end != std::string::npos && end > start) {
            return algorithm.substr(start, end - start);
        }
    }

    const std::string mpir_prefix = "MPIR_";
    if (algorithm.find(mpir_prefix) == 0) {
        const std::size_t start = mpir_prefix.size();
        const char *markers[] = { "_intra_", "_inter_", "_allcomm_", "_sched_" };
        for (const char *marker : markers) {
            const std::size_t end = algorithm.find(marker, start);
            if (end != std::string::npos && end > start) {
                return algorithm.substr(start, end - start);
            }
        }
        if (algorithm == "MPIR_Coll_nb") {
            return "Coll";
        }
    }

    const std::string posix_prefix = "MPIDI_POSIX_mpi_";
    if (algorithm.find(posix_prefix) == 0) {
        const std::size_t start = posix_prefix.size();
        const std::size_t end = algorithm.find('_', start);
        if (end != std::string::npos && end > start) {
            return algorithm.substr(start, end - start);
        }
    }

    return "unknown";
}

bool is_bcast_counter_name(const std::string &name)
{
    return name.find("Bcast") != std::string::npos ||
           name.find("bcast") != std::string::npos ||
           name.find("Ibcast") != std::string::npos ||
           name.find("ibcast") != std::string::npos ||
           name == "MPIR_Coll_nb";
}

bool parse_counter_dump_line(const std::string &line, long long &count, std::string &name)
{
    std::istringstream stream(line);
    if (!(stream >> count >> name)) {
        return false;
    }
    return count > 0 && !name.empty();
}

int find_cvar_index(const char *target_name)
{
    int num_cvars = 0;
    if (MPI_T_cvar_get_num(&num_cvars) != MPI_SUCCESS) {
        return -1;
    }

    for (int i = 0; i < num_cvars; ++i) {
        char name[MPI_MAX_OBJECT_NAME] = { 0 };
        int name_len = MPI_MAX_OBJECT_NAME;
        int verbosity = 0;
        MPI_Datatype datatype = MPI_DATATYPE_NULL;
        MPI_T_enum enumtype = MPI_T_ENUM_NULL;
        char desc[1] = { 0 };
        int desc_len = 1;
        int bind = 0;
        int scope = 0;

        const int rc = MPI_T_cvar_get_info(i, name, &name_len, &verbosity, &datatype, &enumtype,
                                           desc, &desc_len, &bind, &scope);
        if (rc == MPI_SUCCESS && std::strcmp(name, target_name) == 0) {
            return i;
        }
    }

    return -1;
}

bool request_counter_dump(MPI_Comm comm, bool mpi_t_initialized)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (!mpi_t_initialized) {
        if (rank == kRoot) {
            std::cerr << "Could not initialize MPI_T; MPICH collective algorithm counters "
                      << "will not be dumped.\n";
        }
        return false;
    }

    const int cvar_index = find_cvar_index("MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS");
    if (cvar_index < 0) {
        if (rank == kRoot) {
            std::cerr << "Could not find MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS; "
                      << "MPICH collective algorithm counters will not be dumped.\n";
        }
        return false;
    }

    MPI_T_cvar_handle handle = nullptr;
    int count = 0;
    const int alloc_rc = MPI_T_cvar_handle_alloc(cvar_index, nullptr, &handle, &count);
    if (alloc_rc != MPI_SUCCESS) {
        if (rank == kRoot) {
            std::cerr << "MPI_T_cvar_handle_alloc failed for "
                      << "MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS, rc=" << alloc_rc << "\n";
        }
        return false;
    }

    int dump_rank = kRoot;
    const int write_rc = MPI_T_cvar_write(handle, &dump_rank);
    if (write_rc != MPI_SUCCESS) {
        if (rank == kRoot) {
            std::cerr << "MPI_T_cvar_write failed for MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS, "
                      << "rc=" << write_rc << "\n";
        }
        MPI_T_cvar_handle_free(&handle);
        return false;
    }

    MPI_T_cvar_handle_free(&handle);
    return true;
}

std::vector<CounterDumpRow> read_bcast_counter_dump(const std::string &capture_path,
                                                    const Config &config,
                                                    int nproc)
{
    std::vector<CounterDumpRow> rows;
    std::ifstream capture(capture_path.c_str());
    if (!capture) {
        std::cerr << "Could not read captured MPICH collective counter dump: "
                  << capture_path << "\n";
        return rows;
    }

    std::string line;
    while (std::getline(capture, line)) {
        long long count = 0;
        std::string internal_algorithm;
        if (!parse_counter_dump_line(line, count, internal_algorithm) ||
            !is_bcast_counter_name(internal_algorithm)) {
            continue;
        }

        CounterDumpRow row;
        row.requested_collective = "MPI_Bcast";
        row.requested_algorithm = config.algorithm_label;
        row.internal_collective = infer_internal_collective(internal_algorithm);
        row.internal_algorithm = internal_algorithm;
        row.nproc = nproc;
        row.dump_rank = kRoot;
        row.count = count;
        row.raw_line = line;
        rows.push_back(row);
    }

    return rows;
}

bool write_counter_dump_csv(const std::string &path, bool append_output,
                            const std::vector<CounterDumpRow> &rows)
{
    const bool write_header = !append_output || !file_exists(path);
    std::ofstream counter_csv(path.c_str(),
                              append_output ? (std::ios::out | std::ios::app)
                                            : (std::ios::out | std::ios::trunc));
    if (!counter_csv.is_open()) {
        std::cerr << "failed to open counter output file: " << path << "\n";
        return false;
    }

    if (write_header) {
        counter_csv << "requested_collective,requested_algorithm,internal_collective,"
                       "internal_algorithm,nproc,dump_rank,count,raw_line\n";
    }

    for (const CounterDumpRow &row : rows) {
        counter_csv << csv_field(row.requested_collective) << ','
                    << csv_field(row.requested_algorithm) << ','
                    << csv_field(row.internal_collective) << ','
                    << csv_field(row.internal_algorithm) << ','
                    << row.nproc << ','
                    << row.dump_rank << ','
                    << row.count << ','
                    << csv_field(row.raw_line) << '\n';
    }

    counter_csv.close();
    if (!counter_csv) {
        std::cerr << "failed while writing counter output file: " << path << "\n";
        return false;
    }

    std::cout << "Wrote " << path << " (" << rows.size() << " rows)\n";
    return true;
}

int finalize_with_counter_dump_capture(MPI_Comm comm, bool mpi_t_initialized,
                                       std::string &capture_path)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    request_counter_dump(comm, mpi_t_initialized);

    if (rank != kRoot) {
        return MPI_Finalize();
    }

    char path_template[] = "/tmp/mpich_bcast_bench_counters_XXXXXX";
    const int capture_fd = mkstemp(path_template);
    if (capture_fd < 0) {
        std::cerr << "Could not create temporary file for MPICH counter dump capture.\n";
        return MPI_Finalize();
    }
    capture_path = path_template;

    std::cout.flush();
    std::fflush(stdout);

    const int saved_stdout = dup(STDOUT_FILENO);
    if (saved_stdout < 0 || dup2(capture_fd, STDOUT_FILENO) < 0) {
        std::cerr << "Could not redirect stdout for MPICH counter dump capture.\n";
        close(capture_fd);
        if (saved_stdout >= 0) {
            close(saved_stdout);
        }
        std::remove(capture_path.c_str());
        capture_path.clear();
        return MPI_Finalize();
    }
    close(capture_fd);

    const int finalize_rc = MPI_Finalize();

    std::fflush(stdout);
    dup2(saved_stdout, STDOUT_FILENO);
    close(saved_stdout);

    return finalize_rc;
}

/* Convert MPI return codes into readable diagnostics. */
std::string mpi_error_string(int error)
{
    char buffer[MPI_MAX_ERROR_STRING] = { 0 };
    int length = 0;
    if (MPI_Error_string(error, buffer, &length) != MPI_SUCCESS) {
        return "MPI error code " + std::to_string(error);
    }
    return std::string(buffer, length);
}

/*
 * Deterministic payload generator.
 *
 * The input includes algorithm, message size, iteration, and byte offset so a
 * stale or mismatched receive buffer is likely to fail the correctness check.
 */
std::uint8_t payload_byte(int algorithm_index, int message_size, int iteration, int offset)
{
    std::uint32_t value = 0x9e3779b9u;
    value ^= static_cast<std::uint32_t>((algorithm_index + 1) * 0x45d9f3bu);
    value ^= static_cast<std::uint32_t>((message_size + 7) * 0x27d4eb2du);
    value ^= static_cast<std::uint32_t>((iteration + 3) * 0x119de1f3u);
    value ^= static_cast<std::uint32_t>((offset + 11) * 0x3449u);
    return static_cast<std::uint8_t>(value & 0xffu);
}

/*
 * Optional launcher sanity check.
 *
 * On some systems a launcher can start N processes that each become singleton
 * MPI jobs. BCAST_EXPECTED_RANKS catches that before the benchmark writes
 * duplicate rank-0 CSV files.
 */
int expected_comm_size_from_env()
{
    const char *expected = std::getenv("BCAST_EXPECTED_RANKS");
    if (expected == nullptr || expected[0] == '\0') {
        return 0;
    }
    return std::atoi(expected);
}

std::string getenv_string(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::string();
    }
    return value;
}

bool file_exists(const std::string &path)
{
    std::ifstream file(path.c_str());
    return file.good();
}

std::string default_counter_output_path(const std::string &output_path)
{
    const std::string suffix = "_collective_counts";
    const std::string::size_type slash = output_path.find_last_of("/\\");
    const std::string::size_type dot = output_path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        return output_path.substr(0, dot) + suffix + output_path.substr(dot);
    }
    return output_path + suffix + ".csv";
}

}  // namespace

int main(int argc, char **argv)
{
    int mpi_t_provided = 0;
    const bool mpi_t_initialized =
        MPI_T_init_thread(MPI_THREAD_SINGLE, &mpi_t_provided) == MPI_SUCCESS;

    int error = MPI_Init(&argc, &argv);
    if (error != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed: " << mpi_error_string(error) << std::endl;
        if (mpi_t_initialized) {
            MPI_T_finalize();
        }
        return EXIT_FAILURE;
    }

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    /* Abort the whole MPI job with a rank-qualified message. */
    auto fail = [&](const std::string &message, int error_code = EXIT_FAILURE) -> void {
        std::cerr << "Rank " << rank << ": " << message << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error_code);
        std::exit(error_code);
    };

    /* Route all MPI return-code checks through the same failure path. */
    auto mpi_check = [&](int mpi_errno, const std::string &context) -> void {
        if (mpi_errno != MPI_SUCCESS) {
            fail(context + ": " + mpi_error_string(mpi_errno), mpi_errno);
        }
    };

    /* Fail early if the launcher did not form the communicator size we expected. */
    const int expected_comm_size = expected_comm_size_from_env();
    if (expected_comm_size > 0 && world_size != expected_comm_size) {
        if (rank == kRoot) {
            std::cerr << "ERROR: MPI_COMM_WORLD has " << world_size
                      << " rank(s), expected " << expected_comm_size
                      << ". The MPI launcher started processes, but this MPICH "
                      << "library did not connect them into one MPI job. Set "
                      << "BCAST_EXPECTED_RANKS to catch launcher/library mismatches.\n";
        }
        if (mpi_t_initialized) {
            MPI_T_finalize();
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Strict integer parsing so malformed CLI values fail before benchmarking. */
    auto parse_int = [&](const std::string &name, const std::string &value) -> int {
        std::size_t consumed = 0;
        long long parsed = 0;
        try {
            parsed = std::stoll(value, &consumed, 10);
        } catch (const std::exception &) {
            throw std::runtime_error("invalid integer for " + name + ": " + value);
        }

        if (consumed != value.size()) {
            throw std::runtime_error("invalid integer for " + name + ": " + value);
        }
        if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
            throw std::runtime_error("integer out of range for " + name + ": " + value);
        }
        return static_cast<int>(parsed);
    };

    /* Human-readable command-line help, including benchmark loop semantics. */
    auto print_usage = [&](std::ostream &os) -> void {
        os << "Usage: " << argv[0] << " [options]\n"
           << "Options:\n"
           << "  --warmup-rounds N      Warmup iterations per algorithm/message size (default: 5)\n"
           << "  --measured-rounds N    Measured iterations per algorithm/message size (default: 20)\n"
           << "  --rounds N             Alias for --measured-rounds\n"
           << "  --min-msg-size N       First message size in bytes (default: 2)\n"
           << "  --max-msg-size N       Final message size in bytes (default: "
           << kDefaultMaxMsgSize << ")\n"
           << "  --output PATH          CSV output path (default: benchmarking/bcast/bcast_bench.csv)\n"
           << "  --counter-output PATH  MPICH collective algorithm counter CSV path (default: output\n"
           << "                         path with _collective_counts before the extension)\n"
           << "  --algorithm-label NAME Label written to the algorithm column. Defaults to\n"
           << "                         BCAST_ALGORITHM_LABEL, then MPIR_CVAR_BCAST_INTRA_ALGORITHM,\n"
           << "                         then auto.\n"
           << "  --append-output        Append rows and write headers only when creating files\n"
           << "  -h, --help             Print this help text and exit\n"
           << "\n"
           << "Execution order: message size -> warmup/measured iteration.\n"
           << "Algorithm selection is controlled by the job script/environment before launch.\n"
           << "Each MPI_Bcast is preceded by MPI_Barrier. Rank 0 stores one CSV row per\n"
           << "collective call and writes all rows at the end of the run. Rank 0 also writes\n"
           << "MPICH collective algorithm counters dumped by rank 0 during MPI_Finalize.\n";
    };

    /* Parse command-line options on every rank so all ranks agree on the run. */
    Config config;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const std::string &option) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for " + option);
                }
                return argv[++i];
            };

            if (arg == "--warmup-rounds") {
                config.warmup_rounds = parse_int(arg, require_value(arg));
            } else if (arg == "--measured-rounds" || arg == "--rounds") {
                config.measured_rounds = parse_int(arg, require_value(arg));
            } else if (arg == "--min-msg-size") {
                config.min_msg_size = parse_int(arg, require_value(arg));
            } else if (arg == "--max-msg-size") {
                config.max_msg_size = parse_int(arg, require_value(arg));
            } else if (arg == "--output") {
                config.output_path = require_value(arg);
            } else if (arg == "--counter-output") {
                config.counter_output_path = require_value(arg);
            } else if (arg == "--algorithm-label") {
                config.algorithm_label = require_value(arg);
            } else if (arg == "--append-output") {
                config.append_output = true;
            } else if (arg == "-h" || arg == "--help") {
                config.show_help = true;
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        }

        if (config.warmup_rounds < 0) {
            throw std::runtime_error("--warmup-rounds must be >= 0");
        }
        if (config.measured_rounds < 0) {
            throw std::runtime_error("--measured-rounds must be >= 0");
        }
        if (config.warmup_rounds == 0 && config.measured_rounds == 0) {
            throw std::runtime_error("at least one of --warmup-rounds or --measured-rounds must be > 0");
        }
        if (config.min_msg_size <= 0) {
            throw std::runtime_error("--min-msg-size must be > 0");
        }
        if (config.max_msg_size < config.min_msg_size) {
            throw std::runtime_error("--max-msg-size must be >= --min-msg-size");
        }
    } catch (const std::exception &ex) {
        if (rank == 0) {
            std::cerr << ex.what() << "\n\n";
            print_usage(std::cerr);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Utility modes exit after MPI setup so launcher issues are still visible. */
    if (config.show_help) {
        if (rank == 0) {
            print_usage(std::cout);
        }
        if (mpi_t_initialized) {
            MPI_T_finalize();
        }
        mpi_check(MPI_Finalize(), "MPI_Finalize");
        return EXIT_SUCCESS;
    }

    if (config.counter_output_path.empty()) {
        config.counter_output_path = default_counter_output_path(config.output_path);
    }
    if (config.algorithm_label.empty()) {
        config.algorithm_label = getenv_string("BCAST_ALGORITHM_LABEL");
    }
    if (config.algorithm_label.empty()) {
        config.algorithm_label = getenv_string("MPIR_CVAR_BCAST_INTRA_ALGORITHM");
    }
    if (config.algorithm_label.empty()) {
        config.algorithm_label = "auto";
    }

    /* Use a fixed size list: each power of two plus the midpoint to the next level. */
    std::vector<int> message_sizes;
    message_sizes.push_back(config.min_msg_size);
    for (const int message_size : kDefaultMessageSizes) {
        if (message_size > config.min_msg_size && message_size < config.max_msg_size) {
            message_sizes.push_back(message_size);
        }
    }
    if (config.max_msg_size != config.min_msg_size) {
        message_sizes.push_back(config.max_msg_size);
    }
    std::sort(message_sizes.begin(), message_sizes.end());
    message_sizes.erase(std::unique(message_sizes.begin(), message_sizes.end()), message_sizes.end());

    /* One reusable byte buffer, sized for the largest configured message. */
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(config.max_msg_size), 0);

    /* Only rank 0 needs storage because only rank 0 writes the final CSV. */
    std::vector<ResultRow> results;
    const int total_iterations = config.warmup_rounds + config.measured_rounds;
    if (rank == 0) {
        results.reserve(message_sizes.size() * static_cast<std::size_t>(total_iterations));
    }

    /*
     * Main benchmark loop:
     *   message size -> warmup/measured iteration
     *
     * Algorithm selection is intentionally outside this program. Set MPICH CVARs
     * in the job script before launching this benchmark.
     */
    for (const int message_size : message_sizes) {
        for (int iteration = 0; iteration < total_iterations; ++iteration) {
            const bool is_warmup = iteration < config.warmup_rounds;
            const int phase_iteration =
                is_warmup ? iteration : iteration - config.warmup_rounds;

            /* Rebuild the active message range before every broadcast. */
            std::fill(buffer.begin(), buffer.begin() + message_size, 0);
            if (rank == kRoot) {
                for (int offset = 0; offset < message_size; ++offset) {
                    buffer[static_cast<std::size_t>(offset)] =
                        payload_byte(0, message_size, iteration, offset);
                }
            }

            /* The requested pre-call barrier makes each timing sample aligned. */
            mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(before MPI_Bcast)");
            const double start = MPI_Wtime();
            error = MPI_Bcast(buffer.data(), message_size, MPI_BYTE, kRoot, MPI_COMM_WORLD);
            const double stop = MPI_Wtime();
            mpi_check(error, "MPI_Bcast(" + config.algorithm_label + ")");

            const double local_elapsed = stop - start;
            double time_sum = 0.0;
            double time_min = 0.0;
            double time_max = 0.0;
            int local_correct = 1;

            /* Verify every rank received the root payload for this call. */
            for (int offset = 0; offset < message_size; ++offset) {
                if (buffer[static_cast<std::size_t>(offset)] !=
                    payload_byte(0, message_size, iteration, offset)) {
                    local_correct = 0;
                    break;
                }
            }
            int global_correct = 0;

            /*
             * Timing reductions:
             *   sum / nproc -> requested average latency
             *   min/max     -> retained for comparison with average latency
             */
            mpi_check(MPI_Reduce(&local_elapsed, &time_sum, 1, MPI_DOUBLE, MPI_SUM, kRoot,
                                 MPI_COMM_WORLD),
                      "MPI_Reduce(sum elapsed)");
            mpi_check(MPI_Reduce(&local_elapsed, &time_min, 1, MPI_DOUBLE, MPI_MIN, kRoot,
                                 MPI_COMM_WORLD),
                      "MPI_Reduce(min elapsed)");
            mpi_check(MPI_Reduce(&local_elapsed, &time_max, 1, MPI_DOUBLE, MPI_MAX, kRoot,
                                 MPI_COMM_WORLD),
                      "MPI_Reduce(max elapsed)");
            mpi_check(MPI_Reduce(&local_correct, &global_correct, 1, MPI_INT, MPI_MIN, kRoot,
                                 MPI_COMM_WORLD),
                      "MPI_Reduce(correct)");

            if (rank == kRoot) {
                /* Store the row now; defer actual file I/O until the sweep finishes. */
                ResultRow row;
                row.phase = is_warmup ? "warmup" : "actual";
                row.phase_iteration = phase_iteration;
                row.global_iteration = iteration;
                row.algorithm = config.algorithm_label;
                row.root = kRoot;
                row.nproc = world_size;
                row.message_size_bytes = message_size;
                row.time_sum_sec = time_sum;
                row.avg_latency_sec = time_sum / static_cast<double>(world_size);
                row.min_time_sec = time_min;
                row.max_time_sec = time_max;
                row.correct = global_correct == 1;
                results.push_back(row);
            }
        }
    }

    /* Rank 0 writes all stored benchmark rows in one CSV pass. */
    if (rank == kRoot) {
        const bool write_header = !config.append_output || !file_exists(config.output_path);
        std::ofstream csv(config.output_path.c_str(),
                          config.append_output ? (std::ios::out | std::ios::app)
                                               : (std::ios::out | std::ios::trunc));
        if (!csv.is_open()) {
            fail("failed to open output file: " + config.output_path);
        }

        if (write_header) {
            csv << "phase,phase_iteration,global_iteration,algorithm,root,nproc,"
                   "message_size_bytes,time_sum_sec,avg_latency_sec,min_time_sec,max_time_sec,"
                   "correct\n";
        }
        csv << std::fixed << std::setprecision(9);
        for (const ResultRow &row : results) {
            csv << row.phase << ','
                << row.phase_iteration << ','
                << row.global_iteration << ','
                << row.algorithm << ','
                << row.root << ','
                << row.nproc << ','
                << row.message_size_bytes << ','
                << row.time_sum_sec << ','
                << row.avg_latency_sec << ','
                << row.min_time_sec << ','
                << row.max_time_sec << ','
                << (row.correct ? 1 : 0) << '\n';
        }

        csv.close();
        if (!csv) {
            fail("failed while writing output file: " + config.output_path);
        }
        std::cout << "Wrote " << config.output_path << " (" << results.size() << " rows)\n";

    }

    std::string counter_capture_path;
    const int finalize_rc =
        finalize_with_counter_dump_capture(MPI_COMM_WORLD, mpi_t_initialized, counter_capture_path);

    int exit_code = finalize_rc == MPI_SUCCESS ? EXIT_SUCCESS : finalize_rc;
    if (rank == kRoot && !counter_capture_path.empty()) {
        const std::vector<CounterDumpRow> counter_rows =
            read_bcast_counter_dump(counter_capture_path, config, world_size);
        if (!write_counter_dump_csv(config.counter_output_path, config.append_output, counter_rows)) {
            exit_code = EXIT_FAILURE;
        }
        std::remove(counter_capture_path.c_str());
    }

    if (mpi_t_initialized) {
        MPI_T_finalize();
    }
    return exit_code;
}
