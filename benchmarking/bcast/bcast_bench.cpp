#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/* Rank 0 is both the broadcast root and the rank that owns CSV output. */
constexpr int kRoot = 0;

/* MPI_T CVARs used to force MPICH onto the requested Bcast implementation. */
constexpr const char *kBcastIntraAlgorithmCvar = "MPIR_CVAR_BCAST_INTRA_ALGORITHM";
constexpr const char *kBcastDeviceCollectiveCvar = "MPIR_CVAR_BCAST_DEVICE_COLLECTIVE";
constexpr const char *kDeviceCollectivesCvar = "MPIR_CVAR_DEVICE_COLLECTIVES";
constexpr const char *kCollectiveFallbackCvar = "MPIR_CVAR_COLLECTIVE_FALLBACK";

/* Command-line configuration and benchmark defaults. */
struct Config {
    int warmup_rounds = 5;
    int measured_rounds = 20;
    int min_msg_size = 2;
    int max_msg_size = 65536;
    std::string output_path = "benchmarking/bcast/bcast_bench.csv";
    std::vector<std::string> algorithms;
    bool list_algorithms = false;
    bool show_help = false;
};

/* User-visible algorithm name accepted by --algorithms and --list-algorithms. */
struct AlgorithmSpec {
    const char *name;
};

/* Algorithm name plus the resolved integer value written to the MPI_T CVAR. */
struct AlgorithmConfig {
    std::string name;
    int cvar_value = 0;
};

/* Small RAII-like record for MPI_T integer CVAR handles. */
struct IntCvar {
    std::string name;
    MPI_T_enum enum_type = MPI_T_ENUM_NULL;
    MPI_T_cvar_handle handle = MPI_T_CVAR_HANDLE_NULL;
    bool opened = false;
};

/* One CSV row. Rank 0 stores these in memory and writes them after the sweep. */
struct ResultRow {
    std::string phase;
    int phase_iteration = 0;
    int global_iteration = 0;
    std::string algorithm;
    int cvar_value = 0;
    int root = kRoot;
    int nproc = 0;
    int message_size_bytes = 0;
    double time_sum_sec = 0.0;
    double avg_latency_sec = 0.0;
    double min_time_sec = 0.0;
    double max_time_sec = 0.0;
    bool correct = false;
};

/* Matches the algorithms exercised by collective_testing/bcast_success_test.cpp. */
const AlgorithmSpec kSupportedAlgorithms[] = {
    { "binomial" },
    { "nb" },
    { "circ_graph" },
    { "smp" },
    { "scatter_recursive_doubling_allgather" },
    { "scatter_ring_allgather" },
    { "pipelined_tree" },
    { "tree" },
    { "release_gather" },
};

/* Convert MPI and MPI_T return codes into readable diagnostics. */
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

}  // namespace

int main(int argc, char **argv)
{
    /*
     * MPI_T must be initialized before querying/writing MPICH CVARs. MPI itself
     * is initialized immediately after so all ranks can parse and fail together.
     */
    int provided = 0;
    int error = MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    if (error != MPI_SUCCESS) {
        std::cerr << "MPI_T_init_thread failed: " << mpi_error_string(error) << std::endl;
        return EXIT_FAILURE;
    }

    error = MPI_Init(&argc, &argv);
    if (error != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed: " << mpi_error_string(error) << std::endl;
        MPI_T_finalize();
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

    /* Route all MPI and MPI_T return-code checks through the same failure path. */
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
        MPI_Finalize();
        MPI_T_finalize();
        return EXIT_FAILURE;
    }

    /* Build the default algorithm list from the supported algorithm table. */
    std::vector<std::string> default_algorithms;
    default_algorithms.reserve(sizeof(kSupportedAlgorithms) / sizeof(kSupportedAlgorithms[0]));
    for (const AlgorithmSpec &algorithm : kSupportedAlgorithms) {
        default_algorithms.push_back(algorithm.name);
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
           << "  --max-msg-size N       Final message size in bytes (default: 65536)\n"
           << "  --output PATH          CSV output path (default: benchmarking/bcast/bcast_bench.csv)\n"
           << "  --algorithms LIST      Comma-separated subset of algorithms to benchmark\n"
           << "                         Default: ";
        for (std::size_t i = 0; i < default_algorithms.size(); ++i) {
            if (i > 0) {
                os << ",";
            }
            os << default_algorithms[i];
        }
        os << "\n"
           << "  --list-algorithms      Print the supported algorithm names and exit\n"
           << "  -h, --help             Print this help text and exit\n"
           << "\n"
           << "Execution order: algorithm -> message size -> warmup/measured iteration.\n"
           << "Each MPI_Bcast is preceded by MPI_Barrier. Rank 0 stores one CSV row per\n"
           << "collective call and writes all rows at the end of the run.\n";
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
            } else if (arg == "--algorithms") {
                std::stringstream ss(require_value(arg));
                std::string item;
                while (std::getline(ss, item, ',')) {
                    if (!item.empty()) {
                        config.algorithms.push_back(item);
                    }
                }
            } else if (arg == "--list-algorithms") {
                config.list_algorithms = true;
            } else if (arg == "-h" || arg == "--help") {
                config.show_help = true;
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        }

        if (config.algorithms.empty()) {
            config.algorithms = default_algorithms;
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
        mpi_check(MPI_T_finalize(), "MPI_T_finalize");
        mpi_check(MPI_Finalize(), "MPI_Finalize");
        return EXIT_SUCCESS;
    }

    if (config.list_algorithms) {
        if (rank == 0) {
            for (const AlgorithmSpec &algorithm : kSupportedAlgorithms) {
                std::cout << algorithm.name << '\n';
            }
        }
        mpi_check(MPI_T_finalize(), "MPI_T_finalize");
        mpi_check(MPI_Finalize(), "MPI_Finalize");
        return EXIT_SUCCESS;
    }

    /* Open an integer MPI_T CVAR. Optional CVARs return an unopened handle. */
    auto open_int_cvar = [&](const std::string &name, bool required) -> IntCvar {
        IntCvar cvar;
        cvar.name = name;

        int index = -1;
        int rc = MPI_T_cvar_get_index(name.c_str(), &index);
        if (rc != MPI_SUCCESS) {
            if (required) {
                fail("MPI_T_cvar_get_index(" + name + "): " + mpi_error_string(rc), rc);
            }
            return cvar;
        }

        char cvar_name[MPI_MAX_OBJECT_NAME] = { 0 };
        int name_len = MPI_MAX_OBJECT_NAME;
        int verbosity = 0;
        int binding = 0;
        int scope = 0;
        int description_length = 0;
        MPI_Datatype datatype = MPI_DATATYPE_NULL;
        mpi_check(MPI_T_cvar_get_info(index, cvar_name, &name_len, &verbosity, &datatype,
                                      &cvar.enum_type, nullptr, &description_length, &binding,
                                      &scope),
                  "MPI_T_cvar_get_info(" + name + ")");

        int count = 0;
        mpi_check(MPI_T_cvar_handle_alloc(index, nullptr, &cvar.handle, &count),
                  "MPI_T_cvar_handle_alloc(" + name + ")");
        if (count != 1) {
            fail("unexpected MPI_T count for " + name + ": " + std::to_string(count));
        }
        cvar.opened = true;
        return cvar;
    };

    /* Free a CVAR handle only if this build exposed and opened it. */
    auto close_int_cvar = [&](IntCvar &cvar) -> void {
        if (cvar.opened) {
            mpi_check(MPI_T_cvar_handle_free(&cvar.handle),
                      "MPI_T_cvar_handle_free(" + cvar.name + ")");
            cvar.opened = false;
            cvar.handle = MPI_T_CVAR_HANDLE_NULL;
        }
    };

    /* Write a value to an opened CVAR; silently ignore unavailable optional CVARs. */
    auto write_int_cvar = [&](const IntCvar &cvar, int value) -> void {
        if (cvar.opened) {
            mpi_check(MPI_T_cvar_write(cvar.handle, &value),
                      "MPI_T_cvar_write(" + cvar.name + ")");
        }
    };

    /*
     * Resolve a symbolic MPI_T enum item to its integer value.
     *
     * Newer MPICH builds expose enum metadata through MPI_T. The fallback table
     * keeps this benchmark usable with builds that expose only raw integers.
     */
    auto enum_value = [&](const IntCvar &cvar, const std::string &item_name) -> int {
        auto fallback_value = [&]() -> int {
            if (cvar.name == kBcastIntraAlgorithmCvar) {
                if (item_name == "auto")
                    return 0;
                if (item_name == "binomial")
                    return 1;
                if (item_name == "nb")
                    return 2;
                if (item_name == "circ_graph")
                    return 3;
                if (item_name == "smp")
                    return 4;
                if (item_name == "scatter_recursive_doubling_allgather")
                    return 5;
                if (item_name == "scatter_ring_allgather")
                    return 6;
                if (item_name == "pipelined_tree")
                    return 7;
                if (item_name == "tree")
                    return 8;
                if (item_name == "release_gather")
                    return 9;
            } else if (cvar.name == kDeviceCollectivesCvar) {
                if (item_name == "all")
                    return 0;
                if (item_name == "none")
                    return 1;
                if (item_name == "percoll")
                    return 2;
            } else if (cvar.name == kCollectiveFallbackCvar) {
                if (item_name == "error")
                    return 0;
                if (item_name == "print")
                    return 1;
                if (item_name == "silent")
                    return 2;
            }

            fail("enum value '" + item_name + "' not found for " + cvar.name);
            return -1;
        };

        if (cvar.enum_type == MPI_T_ENUM_NULL) {
            return fallback_value();
        }

        int count = 0;
        char enum_name[MPI_MAX_OBJECT_NAME] = { 0 };
        int enum_name_len = MPI_MAX_OBJECT_NAME;
        mpi_check(MPI_T_enum_get_info(cvar.enum_type, &count, enum_name, &enum_name_len),
                  "MPI_T_enum_get_info(" + cvar.name + ")");

        for (int i = 0; i < count; ++i) {
            int value = 0;
            char value_name[MPI_MAX_OBJECT_NAME] = { 0 };
            int value_name_len = MPI_MAX_OBJECT_NAME;
            mpi_check(MPI_T_enum_get_item(cvar.enum_type, i, &value, value_name, &value_name_len),
                      "MPI_T_enum_get_item(" + cvar.name + ")");
            if (item_name == value_name) {
                return value;
            }
        }

        return fallback_value();
    };

    /* Required algorithm selector plus optional CVARs that avoid device overrides. */
    IntCvar bcast_algorithm_cvar = open_int_cvar(kBcastIntraAlgorithmCvar, true);
    IntCvar bcast_device_collective_cvar = open_int_cvar(kBcastDeviceCollectiveCvar, false);
    IntCvar device_collectives_cvar = open_int_cvar(kDeviceCollectivesCvar, false);
    IntCvar collective_fallback_cvar = open_int_cvar(kCollectiveFallbackCvar, false);

    /*
     * Force the benchmark through the MPIR Bcast path. Without these writes,
     * device collectives may bypass MPIR_CVAR_BCAST_INTRA_ALGORITHM.
     */
    write_int_cvar(bcast_device_collective_cvar, 0);
    if (device_collectives_cvar.opened) {
        write_int_cvar(device_collectives_cvar, enum_value(device_collectives_cvar, "none"));
    }
    if (collective_fallback_cvar.opened) {
        write_int_cvar(collective_fallback_cvar, enum_value(collective_fallback_cvar, "error"));
    }

    /* Validate the requested algorithms and resolve their CVAR values once. */
    std::vector<AlgorithmConfig> selected_algorithms;
    selected_algorithms.reserve(config.algorithms.size());
    for (const std::string &name : config.algorithms) {
        auto duplicate = std::find_if(selected_algorithms.begin(), selected_algorithms.end(),
                                      [&](const AlgorithmConfig &algorithm) {
                                          return algorithm.name == name;
                                      });
        if (duplicate != selected_algorithms.end()) {
            continue;
        }

        auto supported = std::find_if(std::begin(kSupportedAlgorithms),
                                      std::end(kSupportedAlgorithms),
                                      [&](const AlgorithmSpec &algorithm) {
                                          return name == algorithm.name;
                                      });
        if (supported == std::end(kSupportedAlgorithms)) {
            fail("unsupported algorithm '" + name +
                 "'. Use --list-algorithms to see the accepted names.");
        }

        AlgorithmConfig algorithm;
        algorithm.name = name;
        algorithm.cvar_value = enum_value(bcast_algorithm_cvar, name);
        selected_algorithms.push_back(algorithm);
    }

    /* Generate 2, 4, 8, ... message sizes and include max_msg_size exactly. */
    std::vector<int> message_sizes;
    std::int64_t current = config.min_msg_size;
    while (current <= config.max_msg_size) {
        message_sizes.push_back(static_cast<int>(current));
        if (current > std::numeric_limits<std::int64_t>::max() / 2) {
            break;
        }
        current *= 2;
    }
    if (message_sizes.empty() || message_sizes.back() != config.max_msg_size) {
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
        results.reserve(selected_algorithms.size() * message_sizes.size() *
                        static_cast<std::size_t>(total_iterations));
    }

    /*
     * Main benchmark loop:
     *   algorithm -> message size -> warmup/measured iteration
     *
     * The algorithm CVAR is written once per algorithm, outside the message and
     * iteration loops, so per-call timing is not polluted by MPI_T writes.
     */
    for (std::size_t algorithm_index = 0; algorithm_index < selected_algorithms.size();
         ++algorithm_index) {
        const AlgorithmConfig &algorithm = selected_algorithms[algorithm_index];
        write_int_cvar(bcast_algorithm_cvar, algorithm.cvar_value);
        mpi_check(MPI_Barrier(MPI_COMM_WORLD),
                  "MPI_Barrier(after selecting " + algorithm.name + ")");

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
                            payload_byte(static_cast<int>(algorithm_index), message_size, iteration,
                                         offset);
                    }
                }

                /* The requested pre-call barrier makes each timing sample aligned. */
                mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(before MPI_Bcast)");
                const double start = MPI_Wtime();
                error = MPI_Bcast(buffer.data(), message_size, MPI_BYTE, kRoot, MPI_COMM_WORLD);
                const double stop = MPI_Wtime();
                mpi_check(error, "MPI_Bcast(" + algorithm.name + ")");

                const double local_elapsed = stop - start;
                double time_sum = 0.0;
                double time_min = 0.0;
                double time_max = 0.0;
                int local_correct = 1;

                /* Verify every rank received the root payload for this call. */
                for (int offset = 0; offset < message_size; ++offset) {
                    if (buffer[static_cast<std::size_t>(offset)] !=
                        payload_byte(static_cast<int>(algorithm_index), message_size, iteration,
                                     offset)) {
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
                    row.algorithm = algorithm.name;
                    row.cvar_value = algorithm.cvar_value;
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
    }

    /* Rank 0 writes all stored benchmark rows in one CSV pass. */
    if (rank == kRoot) {
        std::ofstream csv(config.output_path.c_str(), std::ios::out | std::ios::trunc);
        if (!csv.is_open()) {
            fail("failed to open output file: " + config.output_path);
        }

        csv << "phase,phase_iteration,global_iteration,algorithm,cvar_value,root,nproc,"
               "message_size_bytes,time_sum_sec,avg_latency_sec,min_time_sec,max_time_sec,"
               "correct\n";
        csv << std::fixed << std::setprecision(9);
        for (const ResultRow &row : results) {
            csv << row.phase << ','
                << row.phase_iteration << ','
                << row.global_iteration << ','
                << row.algorithm << ','
                << row.cvar_value << ','
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

    /* Clean up MPI_T handles before finalizing MPI_T and MPI. */
    close_int_cvar(collective_fallback_cvar);
    close_int_cvar(device_collectives_cvar);
    close_int_cvar(bcast_device_collective_cvar);
    close_int_cvar(bcast_algorithm_cvar);

    mpi_check(MPI_T_finalize(), "MPI_T_finalize");
    mpi_check(MPI_Finalize(), "MPI_Finalize");
    return EXIT_SUCCESS;
}
