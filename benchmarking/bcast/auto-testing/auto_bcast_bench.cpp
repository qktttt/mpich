#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDefaultRoot = 0;
constexpr int kDefaultIterations = 100;
constexpr int kDefaultWarmupIterations = 0;
constexpr std::size_t kDefaultMinMsgSize = 32;
constexpr std::size_t kDefaultMaxMsgSize = 32 * 1024 * 1024;

struct Config {
    int iterations = kDefaultIterations;
    int warmup_iterations = kDefaultWarmupIterations;
    int root = kDefaultRoot;
    std::size_t min_msg_size = kDefaultMinMsgSize;
    std::size_t max_msg_size = kDefaultMaxMsgSize;
    std::string output_path = "auto_bcast_bench.csv";
    bool append_output = false;
    bool show_help = false;
};

bool file_exists(const std::string &path)
{
    std::ifstream in(path.c_str());
    return static_cast<bool>(in);
}

int parse_int(const char *value, const char *name)
{
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        std::ostringstream msg;
        msg << "Invalid integer for " << name << ": " << value;
        throw std::runtime_error(msg.str());
    }
    return static_cast<int>(parsed);
}

std::size_t parse_size(const char *value, const char *name)
{
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed == 0) {
        std::ostringstream msg;
        msg << "Invalid byte size for " << name << ": " << value;
        throw std::runtime_error(msg.str());
    }
    return static_cast<std::size_t>(parsed);
}

Config parse_args(int argc, char **argv)
{
    Config config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::ostringstream msg;
                msg << "Missing value for " << name;
                throw std::runtime_error(msg.str());
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            config.show_help = true;
        } else if (arg == "--iterations") {
            config.iterations = parse_int(require_value("--iterations"), "--iterations");
        } else if (arg == "--warmup") {
            config.warmup_iterations = parse_int(require_value("--warmup"), "--warmup");
        } else if (arg == "--root") {
            config.root = parse_int(require_value("--root"), "--root");
        } else if (arg == "--min-msg-size") {
            config.min_msg_size = parse_size(require_value("--min-msg-size"), "--min-msg-size");
        } else if (arg == "--max-msg-size") {
            config.max_msg_size = parse_size(require_value("--max-msg-size"), "--max-msg-size");
        } else if (arg == "--output") {
            config.output_path = require_value("--output");
        } else if (arg == "--append-output") {
            config.append_output = true;
        } else {
            std::ostringstream msg;
            msg << "Unknown argument: " << arg;
            throw std::runtime_error(msg.str());
        }
    }

    if (config.iterations <= 0) {
        throw std::runtime_error("--iterations must be greater than zero");
    }
    if (config.min_msg_size > config.max_msg_size) {
        throw std::runtime_error("--min-msg-size must be <= --max-msg-size");
    }
    if (config.max_msg_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--max-msg-size must fit in an MPI int count");
    }

    return config;
}

void print_help(const char *program)
{
    std::cout
        << "Usage: " << program << " [options]\n"
        << "\n"
        << "Benchmark MPI_Bcast with MPICH auto algorithm selection.\n"
        << "\n"
        << "Options:\n"
        << "  --iterations N       Measured iterations per message size (default: 100)\n"
        << "  --warmup N           Unrecorded warmup iterations per message size (default: 0)\n"
        << "  --root R             Broadcast root rank (default: 0)\n"
        << "  --min-msg-size BYTES Minimum message size (default: 32)\n"
        << "  --max-msg-size BYTES Maximum message size (default: 33554432)\n"
        << "  --output PATH        CSV output path (default: auto_bcast_bench.csv)\n"
        << "  --append-output      Append to an existing CSV\n"
        << "  --help, -h           Show this help\n";
}

std::vector<std::size_t> make_message_sizes(std::size_t min_size, std::size_t max_size)
{
    std::vector<std::size_t> sizes;
    for (std::size_t size = min_size; size <= max_size; size *= 2) {
        sizes.push_back(size);
        if (size > max_size / 2) {
            break;
        }
    }
    return sizes;
}

unsigned char expected_byte(std::size_t index, int iteration, std::size_t msg_size)
{
    return static_cast<unsigned char>((index * 131 + iteration * 17 + msg_size) & 0xff);
}

std::vector<std::size_t> validation_offsets(std::size_t msg_size)
{
    std::vector<std::size_t> offsets;
    offsets.push_back(0);
    offsets.push_back(msg_size / 2);
    offsets.push_back(msg_size - 1);

    const std::size_t stride = 1024 * 1024;
    for (std::size_t offset = stride; offset < msg_size; offset += stride) {
        offsets.push_back(offset);
    }

    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    return offsets;
}

void prepare_buffer(std::vector<unsigned char> &buffer, std::size_t msg_size, int iteration,
                    int rank, int root)
{
    if (rank == root) {
        for (std::size_t i = 0; i < msg_size; ++i) {
            buffer[i] = expected_byte(i, iteration, msg_size);
        }
        return;
    }

    for (const std::size_t offset : validation_offsets(msg_size)) {
        buffer[offset] = static_cast<unsigned char>(expected_byte(offset, iteration, msg_size) ^ 0xff);
    }
}

bool validate_buffer(const std::vector<unsigned char> &buffer, std::size_t msg_size, int iteration)
{
    for (const std::size_t offset : validation_offsets(msg_size)) {
        if (buffer[offset] != expected_byte(offset, iteration, msg_size)) {
            return false;
        }
    }
    return true;
}

void check_expected_ranks(MPI_Comm comm)
{
    const char *expected_env = std::getenv("BCAST_EXPECTED_RANKS");
    if (!expected_env || expected_env[0] == '\0') {
        return;
    }

    const int expected = parse_int(expected_env, "BCAST_EXPECTED_RANKS");
    int nproc = 0;
    MPI_Comm_size(comm, &nproc);
    if (nproc != expected) {
        std::ostringstream msg;
        msg << "MPI_COMM_WORLD size is " << nproc
            << ", but BCAST_EXPECTED_RANKS is " << expected;
        throw std::runtime_error(msg.str());
    }
}

} // namespace

int main(int argc, char **argv)
{
    Config config;
    try {
        config = parse_args(argc, argv);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    if (config.show_help) {
        print_help(argv[0]);
        return 0;
    }

    MPI_Init(&argc, &argv);

    int rank = 0;
    int nproc = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    try {
        if (config.root < 0 || config.root >= nproc) {
            throw std::runtime_error("--root must be a valid rank in MPI_COMM_WORLD");
        }
        check_expected_ranks(MPI_COMM_WORLD);
    } catch (const std::exception &ex) {
        if (rank == 0) {
            std::cerr << ex.what() << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const std::vector<std::size_t> message_sizes =
        make_message_sizes(config.min_msg_size, config.max_msg_size);
    std::vector<unsigned char> buffer(config.max_msg_size);

    std::ofstream output;
    if (rank == 0) {
        const bool write_header = !config.append_output || !file_exists(config.output_path);
        output.open(config.output_path.c_str(),
                    config.append_output ? std::ios::app : std::ios::out);
        if (!output) {
            std::cerr << "Could not open output CSV: " << config.output_path << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        output << std::setprecision(17);
        if (write_header) {
            output << "nproc,root,message_size_bytes,iteration,avg_time_sec,max_time_sec,"
                   << "min_time_sec,correct\n";
        }
    }

    for (const std::size_t msg_size : message_sizes) {
        for (int iter = -config.warmup_iterations; iter < config.iterations; ++iter) {
            const int pattern_iteration = iter < 0 ? iter + config.warmup_iterations : iter;
            prepare_buffer(buffer, msg_size, pattern_iteration, rank, config.root);

            MPI_Barrier(MPI_COMM_WORLD);
            const double start = MPI_Wtime();
            MPI_Bcast(buffer.data(), static_cast<int>(msg_size), MPI_UNSIGNED_CHAR,
                      config.root, MPI_COMM_WORLD);
            const double elapsed = MPI_Wtime() - start;

            const int local_correct = validate_buffer(buffer, msg_size, pattern_iteration) ? 1 : 0;
            double sum_time = 0.0;
            double max_time = 0.0;
            double min_time = 0.0;
            int all_correct = 0;

            MPI_Reduce(&elapsed, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&elapsed, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_correct, &all_correct, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

            if (rank == 0 && iter >= 0) {
                output << nproc << ','
                       << config.root << ','
                       << msg_size << ','
                       << iter << ','
                       << (sum_time / static_cast<double>(nproc)) << ','
                       << max_time << ','
                       << min_time << ','
                       << (all_correct ? "true" : "false") << '\n';
            }
        }
    }

    if (rank == 0) {
        output.close();
        std::cout << "Wrote: " << config.output_path << "\n";
    }

    MPI_Finalize();
    return 0;
}
