// HW 2-1: main. BASE FILE: the judge builds with its own copy, so editing yours changes nothing
// there. main() reads the input, calls your run() in hw2-1.cpp, writes the output and prints
// the time. Read it for what run() receives and what run() must leave behind.
#include <mpi.h>

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "graph_gen.hpp"

// ---- run(): the function you write, in hw2-1.cpp --------------------------------------------
// main() calls it once on every process of MPI_COMM_WORLD, after MPI_Init, with the same arguments.
//   p          the graph (see graph_gen.hpp); nothing of it exists yet: generate what you need
//   sources    the S source vertices; run() computes the shortest paths from each
//   results    S entries main allocated; results[s] describes the distances from sources[s]:
//                n_reached    vertices at finite distance from the source (the source included)
//                checksum     sum over those vertices v of dist[v] * (v + 1), modulo 2^64
//                digest1/2    sum of distance_digest(v, dist[v], 0/1) over finite distances
//                sample_dist  the distance of every vertex of sample_vertices(p), in that order,
//                             -1 where unreachable; main has sized it to p.nsample
// When run() returns, rank 0's results must be complete; main() ignores the other ranks' copies.
// run() must return on every process.
void run(const GraphParams &p, const std::vector<uint32_t> &sources, std::vector<SourceResult> &results);

// Input file format (text, one line):
//   scale edgefactor seed scramble nsample S src_1 ... src_S
static void read_input(const std::string &path, GraphParams &p, std::vector<uint32_t> &sources)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    uint64_t hdr[6] = {};
    if (rank == 0) {
        std::ifstream f(path);
        auto fail = [&](const std::string &why) {
            std::cerr << "Invalid input: " << why << " (" << path << ")\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            std::exit(1);
        };
        // Parse before narrowing or allocating. Signs, overflow and partial tokens are errors.
        auto read_uint = [&](const char *name) -> uint64_t {
            std::string token;
            if (!(f >> token)) fail(std::string("missing ") + name);
            uint64_t value = 0;
            const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
            if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size())
                fail(std::string(name) + " must be an unsigned decimal integer");
            return value;
        };
        const char *names[] = {"scale", "edgefactor", "seed", "scramble", "nsample", "S"};
        for (int i = 0; i < 6; ++i) hdr[i] = read_uint(names[i]);
        if (hdr[0] < 1 || hdr[0] > 26) fail("scale must be in [1, 26]");
        if (hdr[1] < 1 || hdr[1] > 256) fail("edgefactor must be in [1, 256]");
        const uint64_t n = uint64_t(1) << hdr[0];
        if (hdr[1] * n > (uint64_t(1) << 30)) fail("edgefactor * N must be at most 2^30");
        if (hdr[3] > 1) fail("scramble must be 0 or 1");
        if (hdr[4] > (hdr[0] <= 16 ? n : 1000)) fail("nsample must be <= N for scale <= 16, otherwise <= 1000");
        if (hdr[5] < 1 || hdr[5] > 32) fail("S must be in [1, 32]");
        sources.resize(static_cast<size_t>(hdr[5]));
        for (auto &src : sources) {
            const uint64_t v = read_uint("source");
            if (v >= n) fail("source must be in [0, N)");
            src = static_cast<uint32_t>(v);
        }
        std::string extra;
        if (f >> extra) fail("extra token after the S sources");
    }
    MPI_Bcast(hdr, 6, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    p.scale = static_cast<int>(hdr[0]);
    p.edgefactor = static_cast<int>(hdr[1]);
    p.seed = hdr[2];
    p.scramble = static_cast<int>(hdr[3]);
    p.nsample = static_cast<int>(hdr[4]);
    sources.resize(static_cast<size_t>(hdr[5]));
    MPI_Bcast(sources.data(), static_cast<int>(hdr[5]), MPI_UINT32_T, 0, MPI_COMM_WORLD);
}

// Threads this process has, from the kernel's own count. The rules allow none beyond what MPI
// itself starts; the judge reads the maximum main prints below.
static int thread_count()
{
    std::ifstream st("/proc/self/status");
    std::string line;
    while (std::getline(st, line))
        if (line.rfind("Threads:", 0) == 0) return std::atoi(line.c_str() + 8);
    return -1;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 3) {
        if (rank == 0) std::cerr << "Usage: ./hw2-1 <input.txt> <output.bin>\n";
        MPI_Finalize();
        return 1;
    }
    const std::string input_path = argv[1], output_path = argv[2];

    GraphParams p{};
    std::vector<uint32_t> sources;
    read_input(input_path, p, sources);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<SourceResult> results(sources.size());
    for (auto &r : results) r.sample_dist.assign((size_t)p.nsample, -1);

    run(p, sources, results);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::steady_clock::now();

    if (rank == 0) {
        /////////////////////////////////////////////////////////////
        // Output section. The judge parses this file.
        //   int32 S
        //   S times: int32 src, int64 n_reached, uint64 checksum, uint64 digest1, uint64 digest2, int32 nsample, int32 dist[nsample]
        std::ofstream ofs(output_path, std::ios::binary);
        if (!ofs) {
            std::cerr << "Failed to open " << output_path << " for writing.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int32_t S = (int32_t)sources.size();
        ofs.write(reinterpret_cast<const char *>(&S), sizeof(S));
        for (size_t s = 0; s < sources.size(); s++) {
            int32_t src = (int32_t)sources[s], ns = p.nsample;
            ofs.write(reinterpret_cast<const char *>(&src), sizeof(src));
            ofs.write(reinterpret_cast<const char *>(&results[s].n_reached), sizeof(int64_t));
            ofs.write(reinterpret_cast<const char *>(&results[s].checksum), sizeof(uint64_t));
            ofs.write(reinterpret_cast<const char *>(&results[s].digest1), sizeof(uint64_t));
            ofs.write(reinterpret_cast<const char *>(&results[s].digest2), sizeof(uint64_t));
            ofs.write(reinterpret_cast<const char *>(&ns), sizeof(ns));
            ofs.write(reinterpret_cast<const char *>(results[s].sample_dist.data()), sizeof(int32_t) * ns);
        }
        ofs.close();
        /////////////////////////////////////////////////////////////
    }

    int my_threads = thread_count(), max_threads = 0;
    MPI_Reduce(&my_threads, &max_threads, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::chrono::duration<double> total = t1 - t0;
        std::cout << "Graph: scale " << p.scale << ", edgefactor " << p.edgefactor << ", seed " << p.seed
                  << ", scramble " << p.scramble << ", sources " << sources.size() << ", processes: " << size << "\n";
        std::cout << "Compute time: " << total.count() << " s\n";
        std::cout << "Threads: " << max_threads << " (max over processes)\n";
    }
    MPI_Finalize();
    return 0;
}
