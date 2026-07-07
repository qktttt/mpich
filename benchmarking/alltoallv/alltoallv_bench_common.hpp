#ifndef ALLTOALLV_BENCH_COMMON_HPP_INCLUDED
#define ALLTOALLV_BENCH_COMMON_HPP_INCLUDED

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

namespace alltoallv_bench {

struct Config {
    int warmup_rounds = 5;
    int measured_rounds = 20;
    int min_msg_size = 2;
    int max_msg_size = 65532;
    std::string output_path = "benchmarking/alltoallv/alltoallv_bench.csv";
    std::vector<std::string> algorithms;
    std::vector<int> hieata_radices;
    std::vector<int> hieata_bthsizes;
    std::vector<int> parata_radices;
    bool list_algorithms = false;
    bool show_help = false;
};

struct AlgorithmSpec {
    const char *name;
    bool enabled_by_default;
    bool enabled_by_default_in_parameter_sweep;
    bool supports_parameter_sweep;
    bool requires_in_place;
};

struct IntCvar {
    std::string name;
    MPI_T_enum enum_type = MPI_T_ENUM_NULL;
    MPI_T_cvar_handle handle = MPI_T_CVAR_HANDLE_NULL;
    bool available = false;
};

struct RunCase {
    std::string algorithm;
    int algorithm_value = 0;
    int hieata_radix = 0;
    int hieata_bthsize = 0;
    int parata_radix = 0;
};

constexpr const char *kAlltoallvIntraAlgorithmCvar = "MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM";
constexpr const char *kHieataRadixCvar = "MPIR_CVAR_ALLTOALLV_HIEATA_RADIX";
constexpr const char *kHieataBthsizeCvar = "MPIR_CVAR_ALLTOALLV_HIEATA_BTHSIZE";
constexpr const char *kParataRadixCvar = "MPIR_CVAR_ALLTOALLV_PARATA_RADIX";
constexpr const char *kDeviceCollectivesCvar = "MPIR_CVAR_DEVICE_COLLECTIVES";
constexpr const char *kAlltoallvDeviceCollectiveCvar = "MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE";
constexpr const char *kIalltoallvDeviceCollectiveCvar = "MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE";
constexpr const char *kCollectiveFallbackCvar = "MPIR_CVAR_COLLECTIVE_FALLBACK";

inline const std::vector<AlgorithmSpec> &supported_algorithms()
{
    static const std::vector<AlgorithmSpec> algorithms = {
        { "nb", true, false, false, false },
        { "pairwise_sendrecv_replace", false, false, false, true },
        { "scattered", true, false, false, false },
        { "hierarchical_bruck", true, true, true, false },
        { "hierarchical_shm_alltoallv", true, true, true, false },
        { "parameterized_bruck", false, false, true, false },
    };
    return algorithms;
}

inline int expected_comm_size_from_env()
{
    const char *expected = std::getenv("ALLTOALLV_EXPECTED_RANKS");
    if (expected == nullptr || expected[0] == '\0') {
        return 0;
    }
    return std::atoi(expected);
}

inline std::vector<std::string> split_csv_strings(const std::string &text)
{
    std::vector<std::string> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            values.push_back(item);
        }
    }
    return values;
}

inline int parse_int_strict(const std::string &name, const std::string &value)
{
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
}

inline std::vector<int> parse_csv_ints(const std::string &name, const std::string &text)
{
    std::vector<int> values;
    for (const std::string &item : split_csv_strings(text)) {
        int value = parse_int_strict(name, item);
        if (value <= 0) {
            throw std::runtime_error(name + " values must be > 0");
        }
        if (std::find(values.begin(), values.end(), value) == values.end()) {
            values.push_back(value);
        }
    }
    if (values.empty()) {
        throw std::runtime_error(name + " must contain at least one value");
    }
    return values;
}

inline void append_default_algorithms(std::vector<std::string> &names, bool parameter_sweep)
{
    for (const AlgorithmSpec &algorithm : supported_algorithms()) {
        if ((parameter_sweep && algorithm.enabled_by_default_in_parameter_sweep) ||
            (!parameter_sweep && algorithm.enabled_by_default)) {
            names.push_back(algorithm.name);
        }
    }
}

inline void print_usage(std::ostream &os, const char *program, bool parameter_sweep)
{
    std::vector<std::string> defaults;
    append_default_algorithms(defaults, parameter_sweep);

    os << "Usage: " << program << " [options]\n"
       << "Options:\n"
       << "  --warmup-rounds N      Warmup iterations per phase (default: 5)\n"
       << "  --measured-rounds N    Measured iterations per phase (default: 20)\n"
       << "  --rounds N             Alias for --measured-rounds\n"
       << "  --min-msg-size N       First total-send message size per rank in bytes (default: 2)\n"
       << "  --max-msg-size N       Final total-send message size per rank in bytes (default: 65532)\n"
       << "  --output PATH          CSV output path\n"
       << "  --algorithms LIST      Comma-separated subset of algorithms to benchmark\n"
       << "                         Default: ";
    for (std::size_t i = 0; i < defaults.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << defaults[i];
    }
    os << "\n";

    if (parameter_sweep) {
        os << "  --hieata-radices LIST  Radices for hierarchical_bruck (default: 2,4,8,16,32)\n"
           << "  --hieata-bthsizes LIST Batch sizes for hierarchical algorithms (default: 1,2,4,8,16,32)\n"
           << "  --parata-radices LIST  Radices for parameterized_bruck (default: 2,4,8,16,32)\n";
    }

    os << "  --list-algorithms      Print supported algorithm names and exit\n"
       << "  -h, --help             Print this help text and exit\n"
       << "\n"
       << "Message sizes are total send bytes per rank. Counts and displacements are\n"
       << "generated as contiguous, ordered alltoallv layouts. The CSV records the\n"
       << "maximum elapsed time across ranks for each collective call.\n";
}

inline Config parse_config(int argc, char **argv, bool parameter_sweep)
{
    Config config;
    if (parameter_sweep) {
        config.output_path = "benchmarking/alltoallv/alltoallv_param_bench.csv";
        config.hieata_radices = { 2, 4, 8, 16, 32 };
        config.hieata_bthsizes = { 1, 2, 4, 8, 16, 32 };
        config.parata_radices = { 2, 4, 8, 16, 32 };
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string &option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + option);
            }
            return argv[++i];
        };

        if (arg == "--warmup-rounds") {
            config.warmup_rounds = parse_int_strict(arg, require_value(arg));
        } else if (arg == "--measured-rounds" || arg == "--rounds") {
            config.measured_rounds = parse_int_strict(arg, require_value(arg));
        } else if (arg == "--min-msg-size") {
            config.min_msg_size = parse_int_strict(arg, require_value(arg));
        } else if (arg == "--max-msg-size") {
            config.max_msg_size = parse_int_strict(arg, require_value(arg));
        } else if (arg == "--output") {
            config.output_path = require_value(arg);
        } else if (arg == "--algorithms") {
            config.algorithms = split_csv_strings(require_value(arg));
        } else if (arg == "--hieata-radices") {
            if (!parameter_sweep) {
                throw std::runtime_error(arg + " is only supported by alltoallv_param_bench");
            }
            config.hieata_radices = parse_csv_ints(arg, require_value(arg));
        } else if (arg == "--hieata-bthsizes") {
            if (!parameter_sweep) {
                throw std::runtime_error(arg + " is only supported by alltoallv_param_bench");
            }
            config.hieata_bthsizes = parse_csv_ints(arg, require_value(arg));
        } else if (arg == "--parata-radices") {
            if (!parameter_sweep) {
                throw std::runtime_error(arg + " is only supported by alltoallv_param_bench");
            }
            config.parata_radices = parse_csv_ints(arg, require_value(arg));
        } else if (arg == "--list-algorithms") {
            config.list_algorithms = true;
        } else if (arg == "-h" || arg == "--help") {
            config.show_help = true;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (config.algorithms.empty()) {
        append_default_algorithms(config.algorithms, parameter_sweep);
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

    return config;
}

template <typename Fail>
inline void mpi_check(int mpi_errno, const std::string &context, Fail fail)
{
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
}

template <typename Fail>
inline IntCvar open_int_cvar(const std::string &name, bool required, Fail fail)
{
    IntCvar cvar;
    cvar.name = name;

    int index = -1;
    int mpi_errno = MPI_T_cvar_get_index(name.c_str(), &index);
    if (mpi_errno != MPI_SUCCESS) {
        if (required) {
            mpi_check(mpi_errno, "MPI_T_cvar_get_index(" + name + ")", fail);
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
                                  &cvar.enum_type, nullptr, &description_length, &binding, &scope),
              "MPI_T_cvar_get_info(" + name + ")", fail);

    int count = 0;
    mpi_check(MPI_T_cvar_handle_alloc(index, nullptr, &cvar.handle, &count),
              "MPI_T_cvar_handle_alloc(" + name + ")", fail);
    if (count != 1) {
        fail("unexpected MPI_T count for " + name + ": " + std::to_string(count), EXIT_FAILURE);
    }
    cvar.available = true;
    return cvar;
}

inline void close_int_cvar(IntCvar &cvar)
{
    if (cvar.handle != MPI_T_CVAR_HANDLE_NULL) {
        MPI_T_cvar_handle_free(&cvar.handle);
        cvar.handle = MPI_T_CVAR_HANDLE_NULL;
    }
    cvar.available = false;
}

template <typename Fail>
inline void write_int_cvar(const IntCvar &cvar, int value, Fail fail)
{
    if (!cvar.available) {
        return;
    }
    mpi_check(MPI_T_cvar_write(cvar.handle, &value), "MPI_T_cvar_write(" + cvar.name + ")", fail);
}

template <typename Fail>
inline int enum_value(const IntCvar &cvar, const std::string &item_name, Fail fail)
{
    auto fallback_value = [&]() -> int {
        if (cvar.name == kAlltoallvIntraAlgorithmCvar) {
            if (item_name == "auto")
                return 0;
            if (item_name == "nb")
                return 1;
            if (item_name == "pairwise_sendrecv_replace")
                return 2;
            if (item_name == "scattered")
                return 3;
            if (item_name == "hierarchical_bruck")
                return 4;
            if (item_name == "hierarchical_shm_alltoallv")
                return 5;
            if (item_name == "parameterized_bruck")
                return 6;
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

        fail("enum value '" + item_name + "' not found for " + cvar.name, EXIT_FAILURE);
        return -1;
    };

    if (cvar.enum_type == MPI_T_ENUM_NULL) {
        return fallback_value();
    }

    int count = 0;
    char enum_name[MPI_MAX_OBJECT_NAME] = { 0 };
    int enum_name_len = MPI_MAX_OBJECT_NAME;
    mpi_check(MPI_T_enum_get_info(cvar.enum_type, &count, enum_name, &enum_name_len),
              "MPI_T_enum_get_info(" + cvar.name + ")", fail);

    for (int i = 0; i < count; ++i) {
        int value = 0;
        char value_name[MPI_MAX_OBJECT_NAME] = { 0 };
        int value_name_len = MPI_MAX_OBJECT_NAME;
        mpi_check(MPI_T_enum_get_item(cvar.enum_type, i, &value, value_name, &value_name_len),
                  "MPI_T_enum_get_item(" + cvar.name + ")", fail);
        if (item_name == value_name) {
            return value;
        }
    }

    return fallback_value();
}

inline std::vector<int> message_sizes(const Config &config)
{
    std::vector<int> sizes;
    for (int size = config.min_msg_size; size <= config.max_msg_size;) {
        sizes.push_back(size);
        if (size > config.max_msg_size / 2) {
            break;
        }
        size *= 2;
    }
    if (sizes.empty() || sizes.back() != config.max_msg_size) {
        sizes.push_back(config.max_msg_size);
    }
    return sizes;
}

inline std::uint8_t expected_byte(int source_rank, int dest_rank, int global_round, std::size_t offset)
{
    return static_cast<std::uint8_t>((source_rank * 131 + dest_rank * 17 + global_round * 29 +
                                     static_cast<int>(offset % 251)) &
                                    0xff);
}

inline void build_counts(int world_size, int total_send_bytes, std::vector<int> &sendcounts,
                         std::vector<int> &sdispls, std::vector<int> &dest_counts,
                         int &send_bytes)
{
    sendcounts.assign(world_size, 0);
    sdispls.assign(world_size, 0);
    dest_counts.assign(world_size, 0);

    int base = total_send_bytes / world_size;
    int remainder = total_send_bytes % world_size;
    send_bytes = 0;
    for (int rank = 0; rank < world_size; ++rank) {
        dest_counts[rank] = base + (rank < remainder ? 1 : 0);
        sendcounts[rank] = dest_counts[rank];
        sdispls[rank] = send_bytes;
        send_bytes += sendcounts[rank];
    }
}

inline void build_recv_layout(int rank, int world_size, const std::vector<int> &dest_counts,
                              std::vector<int> &recvcounts, std::vector<int> &rdispls,
                              int &recv_bytes)
{
    recvcounts.assign(world_size, dest_counts[rank]);
    rdispls.assign(world_size, 0);
    recv_bytes = 0;
    for (int source = 0; source < world_size; ++source) {
        rdispls[source] = recv_bytes;
        recv_bytes += recvcounts[source];
    }
}

inline void fill_sendbuf(int rank, int world_size, const std::vector<int> &sendcounts,
                         const std::vector<int> &sdispls, int global_round,
                         std::vector<std::uint8_t> &sendbuf)
{
    for (int dest = 0; dest < world_size; ++dest) {
        for (int offset = 0; offset < sendcounts[dest]; ++offset) {
            sendbuf[static_cast<std::size_t>(sdispls[dest] + offset)] =
                expected_byte(rank, dest, global_round, static_cast<std::size_t>(offset));
        }
    }
}

inline bool verify_recvbuf(int rank, int world_size, const std::vector<int> &recvcounts,
                           const std::vector<int> &rdispls, int global_round,
                           const std::vector<std::uint8_t> &recvbuf)
{
    for (int source = 0; source < world_size; ++source) {
        for (int offset = 0; offset < recvcounts[source]; ++offset) {
            std::uint8_t expected =
                expected_byte(source, rank, global_round, static_cast<std::size_t>(offset));
            if (recvbuf[static_cast<std::size_t>(rdispls[source] + offset)] != expected) {
                return false;
            }
        }
    }
    return true;
}

template <typename Fail>
inline std::vector<RunCase> build_run_cases(const Config &config, bool parameter_sweep,
                                            const IntCvar &algorithm_cvar, Fail fail)
{
    std::vector<RunCase> cases;
    for (const std::string &name : config.algorithms) {
        auto it = std::find_if(supported_algorithms().begin(), supported_algorithms().end(),
                               [&](const AlgorithmSpec &entry) { return name == entry.name; });
        if (it == supported_algorithms().end()) {
            fail("unsupported algorithm '" + name + "'. Use --list-algorithms to see the accepted names.",
                 EXIT_FAILURE);
        }
        if (it->requires_in_place) {
            fail("algorithm '" + name +
                     "' requires MPI_IN_PLACE, but this benchmark uses normal send/recv buffers.",
                 EXIT_FAILURE);
        }
        if (parameter_sweep && !it->supports_parameter_sweep) {
            fail("algorithm '" + name + "' has no extra parameter sweep in this benchmark.",
                 EXIT_FAILURE);
        }

        int algorithm_value = enum_value(algorithm_cvar, name, fail);
        if (!parameter_sweep) {
            RunCase run_case;
            run_case.algorithm = name;
            run_case.algorithm_value = algorithm_value;
            cases.push_back(run_case);
        } else if (name == "hierarchical_bruck") {
            for (int radix : config.hieata_radices) {
                for (int bthsize : config.hieata_bthsizes) {
                    RunCase run_case;
                    run_case.algorithm = name;
                    run_case.algorithm_value = algorithm_value;
                    run_case.hieata_radix = radix;
                    run_case.hieata_bthsize = bthsize;
                    cases.push_back(run_case);
                }
            }
        } else if (name == "hierarchical_shm_alltoallv") {
            for (int bthsize : config.hieata_bthsizes) {
                RunCase run_case;
                run_case.algorithm = name;
                run_case.algorithm_value = algorithm_value;
                run_case.hieata_bthsize = bthsize;
                cases.push_back(run_case);
            }
        } else if (name == "parameterized_bruck") {
            for (int radix : config.parata_radices) {
                RunCase run_case;
                run_case.algorithm = name;
                run_case.algorithm_value = algorithm_value;
                run_case.parata_radix = radix;
                cases.push_back(run_case);
            }
        }
    }

    return cases;
}

inline int run(int argc, char **argv, bool parameter_sweep)
{
    const int mpi_t_error = []() {
        int provided = 0;
        return MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    }();
    if (mpi_t_error != MPI_SUCCESS) {
        char buffer[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(mpi_t_error, buffer, &length);
        std::cerr << "MPI_T_init_thread failed: " << std::string(buffer, length) << std::endl;
        return EXIT_FAILURE;
    }

    int error = MPI_Init(&argc, &argv);
    if (error != MPI_SUCCESS) {
        char buffer[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, buffer, &length);
        std::cerr << "MPI_Init failed: " << std::string(buffer, length) << std::endl;
        MPI_T_finalize();
        return EXIT_FAILURE;
    }

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    auto fail = [&](const std::string &message, int error_code) -> void {
        std::cerr << "Rank " << rank << ": " << message << std::endl;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Abort(MPI_COMM_WORLD, error_code);
        }
        std::exit(error_code);
    };

    Config config;
    try {
        config = parse_config(argc, argv, parameter_sweep);
    } catch (const std::exception &ex) {
        if (rank == 0) {
            std::cerr << ex.what() << "\n\n";
            print_usage(std::cerr, argv[0], parameter_sweep);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (config.show_help) {
        if (rank == 0) {
            print_usage(std::cout, argv[0], parameter_sweep);
        }
        MPI_T_finalize();
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    if (config.list_algorithms) {
        if (rank == 0) {
            for (const AlgorithmSpec &algorithm : supported_algorithms()) {
                std::cout << algorithm.name;
                if (algorithm.requires_in_place) {
                    std::cout << " (MPI_IN_PLACE only)";
                }
                std::cout << '\n';
            }
        }
        MPI_T_finalize();
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    const int expected_comm_size = expected_comm_size_from_env();
    if (expected_comm_size > 0 && world_size != expected_comm_size) {
        if (rank == 0) {
            std::cerr << "ERROR: MPI_COMM_WORLD has " << world_size << " rank(s), expected "
                      << expected_comm_size << ". Set ALLTOALLV_EXPECTED_RANKS to catch "
                      << "launcher/library mismatches." << std::endl;
        }
        MPI_T_finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm shared_comm = MPI_COMM_NULL;
    mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &shared_comm),
              "MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)", fail);
    int local_size = 0;
    mpi_check(MPI_Comm_size(shared_comm, &local_size), "MPI_Comm_size(shared_comm)", fail);
    mpi_check(MPI_Comm_free(&shared_comm), "MPI_Comm_free(shared_comm)", fail);
    int min_ppn = 0;
    int max_ppn = 0;
    mpi_check(MPI_Allreduce(&local_size, &min_ppn, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
              "MPI_Allreduce(ppn min)", fail);
    mpi_check(MPI_Allreduce(&local_size, &max_ppn, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
              "MPI_Allreduce(ppn max)", fail);
    if (rank == 0 && min_ppn != max_ppn) {
        std::cerr << "Warning: detected non-uniform processes-per-node layout; csv will record "
                  << "the maximum local process count (" << max_ppn << ")." << std::endl;
    }

    IntCvar algorithm_cvar = open_int_cvar(kAlltoallvIntraAlgorithmCvar, true, fail);
    IntCvar device_collectives_cvar = open_int_cvar(kDeviceCollectivesCvar, false, fail);
    IntCvar alltoallv_device_cvar = open_int_cvar(kAlltoallvDeviceCollectiveCvar, false, fail);
    IntCvar ialltoallv_device_cvar = open_int_cvar(kIalltoallvDeviceCollectiveCvar, false, fail);
    IntCvar fallback_cvar = open_int_cvar(kCollectiveFallbackCvar, true, fail);
    IntCvar hieata_radix_cvar = open_int_cvar(kHieataRadixCvar, parameter_sweep, fail);
    IntCvar hieata_bthsize_cvar = open_int_cvar(kHieataBthsizeCvar, parameter_sweep, fail);
    IntCvar parata_radix_cvar = open_int_cvar(kParataRadixCvar, parameter_sweep, fail);

    write_int_cvar(device_collectives_cvar, enum_value(device_collectives_cvar, "none", fail),
                   fail);
    write_int_cvar(alltoallv_device_cvar, 0, fail);
    write_int_cvar(ialltoallv_device_cvar, 0, fail);
    write_int_cvar(fallback_cvar, enum_value(fallback_cvar, "error", fail), fail);

    std::vector<RunCase> cases = build_run_cases(config, parameter_sweep, algorithm_cvar, fail);
    std::vector<int> sizes = message_sizes(config);

    std::ofstream csv;
    if (rank == 0) {
        csv.open(config.output_path.c_str(), std::ios::out | std::ios::trunc);
        if (!csv) {
            fail("unable to open output CSV: " + config.output_path, EXIT_FAILURE);
        }
        csv << "phase,phase_round,global_round,algorithm,nproc,ppn,message_size_bytes,"
            << "send_bytes,recv_bytes,max_time_sec";
        if (parameter_sweep) {
            csv << ",hieata_radix,hieata_bthsize,parata_radix";
        }
        csv << "\n";
        csv << std::setprecision(12);
    }

    int global_round = 0;
    std::vector<int> sendcounts;
    std::vector<int> sdispls;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;
    std::vector<int> dest_counts;
    std::vector<std::uint8_t> sendbuf;
    std::vector<std::uint8_t> recvbuf;

    auto execute_phase = [&](const char *phase_name, int rounds) -> void {
        for (int phase_round = 0; phase_round < rounds; ++phase_round) {
            for (int message_size : sizes) {
                int send_bytes = 0;
                build_counts(world_size, message_size, sendcounts, sdispls, dest_counts,
                             send_bytes);
                int recv_bytes = 0;
                build_recv_layout(rank, world_size, dest_counts, recvcounts, rdispls, recv_bytes);
                sendbuf.assign(static_cast<std::size_t>(std::max(send_bytes, 1)), 0);
                recvbuf.assign(static_cast<std::size_t>(std::max(recv_bytes, 1)), 0);

                for (const RunCase &run_case : cases) {
                    write_int_cvar(algorithm_cvar, run_case.algorithm_value, fail);
                    if (parameter_sweep) {
                        if (run_case.hieata_radix > 0) {
                            write_int_cvar(hieata_radix_cvar, run_case.hieata_radix, fail);
                        }
                        if (run_case.hieata_bthsize > 0) {
                            write_int_cvar(hieata_bthsize_cvar, run_case.hieata_bthsize, fail);
                        }
                        if (run_case.parata_radix > 0) {
                            write_int_cvar(parata_radix_cvar, run_case.parata_radix, fail);
                        }
                    }

                    fill_sendbuf(rank, world_size, sendcounts, sdispls, global_round, sendbuf);
                    std::fill(recvbuf.begin(), recvbuf.end(), 0);
                    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(before MPI_Alltoallv)",
                              fail);
                    double start_time = MPI_Wtime();
                    int alltoallv_error =
                        MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(),
                                      MPI_UNSIGNED_CHAR, recvbuf.data(), recvcounts.data(),
                                      rdispls.data(), MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
                    double elapsed = MPI_Wtime() - start_time;
                    mpi_check(alltoallv_error, "MPI_Alltoallv(" + run_case.algorithm + ")", fail);

                    int local_ok =
                        verify_recvbuf(rank, world_size, recvcounts, rdispls, global_round, recvbuf)
                            ? 1
                            : 0;
                    int global_ok = 0;
                    mpi_check(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                                            MPI_COMM_WORLD),
                              "MPI_Allreduce(verify)", fail);
                    if (global_ok != 1) {
                        fail("verification failed for algorithm " + run_case.algorithm,
                             EXIT_FAILURE);
                    }

                    double max_elapsed = 0.0;
                    mpi_check(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                                         MPI_COMM_WORLD),
                              "MPI_Reduce(max elapsed)", fail);
                    if (rank == 0) {
                        csv << phase_name << ',' << phase_round << ',' << global_round << ','
                            << run_case.algorithm << ',' << world_size << ',' << max_ppn << ','
                            << message_size << ',' << send_bytes << ',' << recv_bytes << ','
                            << max_elapsed;
                        if (parameter_sweep) {
                            csv << ',' << run_case.hieata_radix << ',' << run_case.hieata_bthsize
                                << ',' << run_case.parata_radix;
                        }
                        csv << '\n';
                    }
                    ++global_round;
                }
            }
        }
    };

    execute_phase("warmup", config.warmup_rounds);
    execute_phase("actual", config.measured_rounds);

    if (rank == 0) {
        csv.close();
    }

    close_int_cvar(parata_radix_cvar);
    close_int_cvar(hieata_bthsize_cvar);
    close_int_cvar(hieata_radix_cvar);
    close_int_cvar(fallback_cvar);
    close_int_cvar(ialltoallv_device_cvar);
    close_int_cvar(alltoallv_device_cvar);
    close_int_cvar(device_collectives_cvar);
    close_int_cvar(algorithm_cvar);

    MPI_T_finalize();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

}  // namespace alltoallv_bench

#endif
