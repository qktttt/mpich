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

/*
 * Runtime configuration parsed from the command line.
 *
 * The benchmark has two phases:
 *   1. warmup rounds
 *   2. measured rounds
 *
 * Both phases are executed and both are written to the CSV output.
 */
struct Config {
    int warmup_rounds = 5;
    int measured_rounds = 20;
    int min_msg_size = 2;
    int max_msg_size = 65532;
    std::string output_path = "benchmarking/allreduce/allreduce_bench.csv";
    std::vector<std::string> algorithms;
    bool list_algorithms = false;
    bool show_help = false;
};

struct AlgorithmConfig {
    std::string name;
    int cvar_value;
    bool requires_count_ge_pof2 = false;
};

struct AlgorithmSpec {
    const char *name;
    bool enabled_by_default;
    bool requires_count_ge_pof2;
};

/* These are the MPICH CVARs used to force a specific MPIR-level Allreduce path. */
constexpr const char *kAllreduceIntraAlgorithmCvar = "MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM";
constexpr const char *kDeviceCollectivesCvar = "MPIR_CVAR_DEVICE_COLLECTIVES";
constexpr const char *kAllreduceDeviceCollectiveCvar = "MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE";
constexpr const char *kIallreduceDeviceCollectiveCvar = "MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE";
constexpr const char *kCollectiveFallbackCvar = "MPIR_CVAR_COLLECTIVE_FALLBACK";

int expected_comm_size_from_env()
{
    const char *expected = std::getenv("ALLREDUCE_EXPECTED_RANKS");
    if (expected == nullptr || expected[0] == '\0') {
        return 0;
    }
    return std::atoi(expected);
}

}  // namespace

int main(int argc, char **argv)
{
    /*
     * This benchmark runs a single MPI job and changes the MPICH Allreduce
     * implementation in-process through MPI_T CVARs.
     *
     * Execution order:
     *   round -> message size -> algorithm
     *
     * For each collective call, every rank measures its local elapsed time and
     * rank 0 records the maximum across ranks. That makes each CSV row represent
     * the slowest participant in that specific allreduce.
     */
    const std::vector<AlgorithmSpec> supported_algorithms = {
        { "nb", true, false },
        { "smp", true, false },
        { "recursive_doubling", true, false },
        { "recursive_multiplying", true, false },
        { "reduce_scatter_allgather", true, true },
        { "tree", true, false },
        { "recexch", true, false },
        { "ring", true, false },
        { "k_reduce_scatter_allgather", true, false },
        { "hierarchical", true, false },
        { "ccl", false, false },
    };

    std::vector<std::string> default_algorithms;
    default_algorithms.reserve(supported_algorithms.size());
    for (const AlgorithmSpec &algorithm : supported_algorithms) {
        if (algorithm.enabled_by_default) {
            default_algorithms.push_back(algorithm.name);
        }
    }

    int provided = 0;
    int error = MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    if (error != MPI_SUCCESS) {
        char buffer[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, buffer, &length);
        std::cerr << "MPI_T_init_thread failed: " << std::string(buffer, length) << std::endl;
        return EXIT_FAILURE;
    }

    error = MPI_Init(&argc, &argv);
    if (error != MPI_SUCCESS) {
        char buffer[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, buffer, &length);
        std::cerr << "MPI_Init failed: " << std::string(buffer, length) << std::endl;
        return EXIT_FAILURE;
    }

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    /* Abort the whole MPI job with a rank-qualified error message. */
    auto fail = [&](const std::string &message, int error_code = EXIT_FAILURE) -> void {
        std::cerr << "Rank " << rank << ": " << message << std::endl;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Abort(MPI_COMM_WORLD, error_code);
        }
        std::exit(error_code);
    };

    /* Small helper to route all MPI/MPI_T return-code checks through fail(). */
    auto mpi_check = [&](int mpi_errno, const std::string &context) -> void {
        if (mpi_errno == MPI_SUCCESS) {
            return;
        }

        char buffer[MPI_MAX_ERROR_STRING];
        int length = 0;
        buffer[0] = '\0';
        if (MPI_Error_string(mpi_errno, buffer, &length) != MPI_SUCCESS) {
            fail(context + ": unable to retrieve MPI error string", mpi_errno);
        }
        fail(context + ": " + std::string(buffer, length), mpi_errno);
    };

    const int expected_comm_size = expected_comm_size_from_env();
    if (expected_comm_size > 0 && world_size != expected_comm_size) {
        if (rank == 0) {
            std::cerr << "ERROR: MPI_COMM_WORLD has " << world_size
                      << " rank(s), expected " << expected_comm_size
                      << ". The MPI launcher started processes, but this MPICH "
                      << "library did not connect them into one MPI job. Set "
                      << "ALLREDUCE_EXPECTED_RANKS to catch launcher/library mismatches."
                      << std::endl;
        }
        MPI_T_finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Strict integer parsing so malformed CLI values fail early and clearly. */
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

    /* Print the CLI and document the meaning of the main runtime parameters. */
    auto print_usage = [&](std::ostream &os) -> void {
        os << "Usage: " << argv[0] << " [options]\n"
           << "Options:\n"
           << "  --warmup-rounds N      Warmup iterations per phase (default: 5)\n"
           << "  --measured-rounds N    Measured iterations per phase (default: 20)\n"
           << "  --rounds N             Alias for --measured-rounds\n"
           << "  --min-msg-size N       First message size in bytes (default: 2)\n"
           << "  --max-msg-size N       Final message size in bytes (default: 65532)\n"
           << "  --output PATH          CSV output path (default: benchmarking/allreduce/allreduce_bench.csv)\n"
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
           << "Message sizes are generated by doubling from the minimum size and the maximum\n"
           << "size is appended if it is not already present.\n"
           << "The benchmark uses MPI_UNSIGNED_CHAR with MPI_BXOR so every forced algorithm\n"
           << "sees a built-in, commutative reduction operator.\n"
           << "Rows for reduce_scatter_allgather are skipped when the message size is smaller\n"
           << "than the nearest power-of-two not exceeding the communicator size.\n"
           << "The CSV contains one row per collective call and labels each row as warmup\n"
           << "or actual with the phase column.\n";
    };

    Config config;
    try {
        /* Parse all CLI arguments first, then validate the final configuration. */
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

    /* Early-exit utility modes after MPI is up and arguments are validated. */
    if (config.show_help) {
        if (rank == 0) {
            print_usage(std::cout);
        }
        MPI_T_finalize();
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    if (config.list_algorithms) {
        if (rank == 0) {
            for (const AlgorithmSpec &algorithm : supported_algorithms) {
                std::cout << algorithm.name << '\n';
            }
        }
        MPI_T_finalize();
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    auto nearest_power_of_two_le = [&](int value) -> int {
        int result = 1;
        while (result <= value / 2) {
            result *= 2;
        }
        return result;
    };
    const int pof2 = nearest_power_of_two_le(world_size);

    /*
     * Estimate processes per node with MPI_COMM_TYPE_SHARED. If the job is not
     * perfectly balanced across nodes, keep the maximum local size in the CSV.
     */
    MPI_Comm shared_comm = MPI_COMM_NULL;
    mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &shared_comm),
              "MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)");
    int local_size = 0;
    mpi_check(MPI_Comm_size(shared_comm, &local_size), "MPI_Comm_size(shared_comm)");
    mpi_check(MPI_Comm_free(&shared_comm), "MPI_Comm_free(shared_comm)");

    int min_ppn = 0;
    int max_ppn = 0;
    mpi_check(MPI_Allreduce(&local_size, &min_ppn, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
              "MPI_Allreduce(ppn min)");
    mpi_check(MPI_Allreduce(&local_size, &max_ppn, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
              "MPI_Allreduce(ppn max)");
    if (rank == 0 && min_ppn != max_ppn) {
        std::cerr << "Warning: detected non-uniform processes-per-node layout; csv will record "
                  << "the maximum local process count (" << max_ppn << ")." << std::endl;
    }

    struct IntCvar {
        std::string name;
        MPI_T_enum enum_type = MPI_T_ENUM_NULL;
        MPI_T_cvar_handle handle = MPI_T_CVAR_HANDLE_NULL;
    };

    /*
     * Open one integer CVAR handle once and keep it for the whole run.
     *
     * The benchmark rewrites the Allreduce algorithm CVAR on every iteration,
     * so opening the handle once avoids repeated MPI_T lookup/allocation
     * overhead.
     */
    auto open_int_cvar = [&](const std::string &name) -> IntCvar {
        IntCvar cvar;
        cvar.name = name;

        int index = -1;
        mpi_check(MPI_T_cvar_get_index(name.c_str(), &index), "MPI_T_cvar_get_index(" + name + ")");

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

        return cvar;
    };

    /* Release the corresponding MPI_T handle when the benchmark is done. */
    auto close_int_cvar = [&](IntCvar &cvar) -> void {
        if (cvar.handle != MPI_T_CVAR_HANDLE_NULL) {
            MPI_T_cvar_handle_free(&cvar.handle);
            cvar.handle = MPI_T_CVAR_HANDLE_NULL;
        }
    };

    /* Write a new integer value into a CVAR. */
    auto write_int_cvar = [&](const IntCvar &cvar, int value) -> void {
        mpi_check(MPI_T_cvar_write(cvar.handle, &value), "MPI_T_cvar_write(" + cvar.name + ")");
    };

    /*
     * Convert a symbolic enum name such as "recursive_doubling" to the integer
     * value expected by MPI_T.
     *
     * Some MPICH builds expose the enum metadata through MPI_T and some only
     * expose raw integers, so this helper first tries the MPI_T enum object and
     * then falls back to a hard-coded MPICH mapping.
     */
    auto enum_value = [&](const IntCvar &cvar, const std::string &item_name) -> int {
        auto fallback_value = [&](void) -> int {
            if (cvar.name == kAllreduceIntraAlgorithmCvar) {
                if (item_name == "auto")
                    return 0;
                if (item_name == "nb")
                    return 1;
                if (item_name == "smp")
                    return 2;
                if (item_name == "recursive_doubling")
                    return 3;
                if (item_name == "recursive_multiplying")
                    return 4;
                if (item_name == "reduce_scatter_allgather")
                    return 5;
                if (item_name == "tree")
                    return 6;
                if (item_name == "recexch")
                    return 7;
                if (item_name == "ring")
                    return 8;
                if (item_name == "k_reduce_scatter_allgather")
                    return 9;
                if (item_name == "hierarchical")
                    return 10;
                if (item_name == "ccl")
                    return 11;
                if (item_name == "release_gather")
                    return 12;
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

    IntCvar allreduce_algorithm_cvar = open_int_cvar(kAllreduceIntraAlgorithmCvar);
    IntCvar device_collectives_cvar = open_int_cvar(kDeviceCollectivesCvar);
    IntCvar allreduce_device_collective_cvar = open_int_cvar(kAllreduceDeviceCollectiveCvar);
    IntCvar iallreduce_device_collective_cvar = open_int_cvar(kIallreduceDeviceCollectiveCvar);
    IntCvar collective_fallback_cvar = open_int_cvar(kCollectiveFallbackCvar);

    /*
     * Force the benchmark through the MPIR-level Allreduce path. Otherwise
     * device overrides may bypass the implementation selected by the Allreduce
     * algorithm CVAR.
     */
    write_int_cvar(device_collectives_cvar, enum_value(device_collectives_cvar, "none"));
    write_int_cvar(allreduce_device_collective_cvar, 0);
    write_int_cvar(iallreduce_device_collective_cvar, 0);
    write_int_cvar(collective_fallback_cvar, enum_value(collective_fallback_cvar, "error"));

    /*
     * Resolve the requested algorithm names once up front.
     *
     * After this point the hot loop only writes the already-resolved integer
     * values, rather than repeating enum lookups every time.
     */
    std::vector<AlgorithmConfig> selected_algorithms;
    selected_algorithms.reserve(config.algorithms.size());
    for (const std::string &name : config.algorithms) {
        if (name == "release_gather") {
            fail("algorithm 'release_gather' is intentionally excluded from this benchmark");
        }

        auto existing = std::find_if(selected_algorithms.begin(), selected_algorithms.end(),
                                     [&](const AlgorithmConfig &entry) {
                                         return entry.name == name;
                                     });
        if (existing != selected_algorithms.end()) {
            continue;
        }

        auto it = std::find_if(supported_algorithms.begin(), supported_algorithms.end(),
                               [&](const AlgorithmSpec &entry) { return name == entry.name; });
        if (it == supported_algorithms.end()) {
            fail("unsupported algorithm '" + name +
                 "'. Use --list-algorithms to see the accepted names.");
        }

        AlgorithmConfig selected;
        selected.name = name;
        selected.cvar_value = enum_value(allreduce_algorithm_cvar, name);
        selected.requires_count_ge_pof2 = it->requires_count_ge_pof2;
        selected_algorithms.push_back(selected);
    }

    /* Generate 2x message sizes and force the configured max size into the sweep. */
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

    std::vector<unsigned char> send_buffer(static_cast<std::size_t>(config.max_msg_size));
    std::vector<unsigned char> recv_buffer(static_cast<std::size_t>(config.max_msg_size));
    std::vector<bool> skip_reported(selected_algorithms.size(), false);

    std::ofstream csv;
    if (rank == 0) {
        csv.open(config.output_path, std::ios::out | std::ios::trunc);
        if (!csv.is_open()) {
            fail("failed to open output file: " + config.output_path);
        }

        /*
         * CSV columns:
         *   phase        -> "warmup" or "actual"
         *   phase_round  -> round number within that phase
         *   global_round -> round number in the combined run
         *   algorithm    -> MPICH Allreduce algorithm name
         *   nproc, ppn   -> global process count and max processes per node
         *   message_size_bytes
         *   max_time_sec -> slowest-rank elapsed time for this collective call
         */
        csv << "phase,phase_round,global_round,algorithm,nproc,ppn,message_size_bytes,max_time_sec\n";
        csv << std::fixed << std::setprecision(9);
    }

    /*
     * Benchmark order:
     *   outer  -> round
     *   middle -> message size
     *   inner  -> algorithm
     *
     * Each row records the slowest-rank time for one collective call. Warmup
     * and measured rounds are both written to the CSV and distinguished by the
     * phase column.
     */
    const int total_rounds = config.warmup_rounds + config.measured_rounds;
    for (int round = 0; round < total_rounds; ++round) {
        const bool is_warmup = round < config.warmup_rounds;
        const char *phase = is_warmup ? "warmup" : "actual";
        const int phase_round = is_warmup ? round : round - config.warmup_rounds;

        for (std::size_t message_size_index = 0; message_size_index < message_sizes.size();
             ++message_size_index) {
            const int message_size = message_sizes[message_size_index];

            for (std::size_t algorithm_index = 0; algorithm_index < selected_algorithms.size();
                 ++algorithm_index) {
                const AlgorithmConfig &algorithm = selected_algorithms[algorithm_index];

                if (algorithm.requires_count_ge_pof2 && message_size < pof2) {
                    if (rank == 0 && !skip_reported[algorithm_index]) {
                        std::cerr << "Skipping algorithm " << algorithm.name
                                  << " for message sizes below " << pof2
                                  << " bytes because this forced MPICH path requires count >= pof2."
                                  << std::endl;
                        skip_reported[algorithm_index] = true;
                    }
                    continue;
                }

                /* Switch MPICH to the target Allreduce implementation for this call. */
                write_int_cvar(allreduce_algorithm_cvar, algorithm.cvar_value);

                /* Reinitialize only the active range to avoid reading stale bytes. */
                for (int i = 0; i < message_size; ++i) {
                    send_buffer[static_cast<std::size_t>(i)] =
                        static_cast<unsigned char>((rank + round + algorithm_index + i) & 0xff);
                }
                std::fill(recv_buffer.begin(), recv_buffer.begin() + message_size, 0);

                /*
                 * A barrier before and after the allreduce makes each sample a
                 * clean per-call timing and avoids overlap with the previous/next
                 * test case.
                 */
                mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(before MPI_Allreduce)");
                const double start = MPI_Wtime();
                error = MPI_Allreduce(send_buffer.data(), recv_buffer.data(), message_size,
                                      MPI_UNSIGNED_CHAR, MPI_BXOR, MPI_COMM_WORLD);
                const double stop = MPI_Wtime();
                mpi_check(error, "MPI_Allreduce(" + algorithm.name + ")");
                mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(after MPI_Allreduce)");

                const double local_elapsed = stop - start;
                double max_elapsed = 0.0;
                mpi_check(MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                                     MPI_COMM_WORLD),
                          "MPI_Reduce(max elapsed)");

                /* Only rank 0 writes the CSV row, using the slowest-rank timing. */
                if (rank == 0) {
                    csv << phase << ','
                        << phase_round << ','
                        << round << ','
                        << algorithm.name << ','
                        << world_size << ','
                        << max_ppn << ','
                        << message_size << ','
                        << max_elapsed << '\n';
                }
            }
        }
    }

    if (rank == 0) {
        csv.close();
    }

    close_int_cvar(collective_fallback_cvar);
    close_int_cvar(iallreduce_device_collective_cvar);
    close_int_cvar(allreduce_device_collective_cvar);
    close_int_cvar(device_collectives_cvar);
    close_int_cvar(allreduce_algorithm_cvar);

    mpi_check(MPI_T_finalize(), "MPI_T_finalize");
    mpi_check(MPI_Finalize(), "MPI_Finalize");
    return EXIT_SUCCESS;
}
