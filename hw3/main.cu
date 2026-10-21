// HW 3-1: main. FIXED FILE: the judge builds with its own copy, so editing yours changes nothing
// there. It reads the test case, builds the first generation, calls your run() in hw3-1.cu,
// hashes the last generation into the output and prints the time. Read it to see what run() is
// given and what it must leave behind; ca.hpp has the rule itself.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ca.hpp"

// Input file format (text; anything after a '#' on a line is a comment):
//
//   W H
//   R<R>,C<C>,M<0|1>,S<lo>..<hi>,B<lo>..<hi>,N<M|N>    the rule
//   torus | dead                                       the boundary
//   T                                                  generations to run
//   n                                                  then n seed rectangles:
//   x y w h p seed                                     cells alive with probability p%
//
// The first generation is all dead except inside the seed rectangles. Cell (x, y) of a rectangle
// is alive when cell_hash(seed, x, y) % 1000000 < p * 10000; rectangles may overlap, and a cell
// alive in any of them is alive. Each rectangle lies inside the grid, and 0 <= p <= 100.

constexpr int MAX_SIDE = 32768, MAX_STEPS = 5000;

struct Seed { int x, y, w, h, p; uint64_t seed; };
struct Case { Problem p; std::vector<Seed> seeds; };

[[noreturn]] static void die(const std::string &msg)
{
    std::cerr << "hw3-1: " << msg << "\n";
    std::exit(1);
}

static void cuda_check(cudaError_t e, const char *what)
{
    if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}

static uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static uint64_t cell_hash(uint64_t seed, int x, int y)
{
    return splitmix64(splitmix64(seed) ^ (uint64_t(uint32_t(y)) << 32 | uint32_t(x)));
}

static Rule parse_rule(const std::string &s)
{
    Rule r{};
    int m, end = 0;
    char shape;
    if (std::sscanf(s.c_str(), "R%d,C%d,M%d,S%d..%d,B%d..%d,N%c%n", &r.radius, &r.states, &m, &r.survive_lo,
                    &r.survive_hi, &r.birth_lo, &r.birth_hi, &shape, &end) != 8 ||
        size_t(end) != s.size() || (m != 0 && m != 1) || (shape != 'M' && shape != 'N'))
        die("rule: cannot read " + s);
    r.count_centre = m == 1;
    r.shape = shape == 'N' ? Shape::VonNeumann : Shape::Moore;
    if (r.radius < 1 || r.radius > MAX_RADIUS || r.states < 2 || r.states > MAX_STATES)
        die("rule: R or C out of range in " + s);
    // n counts at most maxn neighbours, plus the cell itself under M1.
    const int R = r.radius, maxn = r.shape == Shape::Moore ? (2 * R + 1) * (2 * R + 1) - 1 : 2 * R * (R + 1);
    if (r.survive_lo < m || r.survive_lo > r.survive_hi || r.survive_hi > maxn + m || r.birth_lo < 1 ||
        r.birth_lo > r.birth_hi || r.birth_hi > maxn + m)
        die("rule: S or B out of range in " + s);
    return r;
}

static Case read_case(const char *path)
{
    std::ifstream f(path);
    if (!f) die(std::string("cannot open ") + path);
    std::stringstream in;
    for (std::string line; std::getline(f, line);) in << line.substr(0, line.find('#')) << "\n";

    Case c{};
    Problem &p = c.p;
    std::string rule, boundary;
    size_t n;
    if (!(in >> p.width >> p.height >> rule >> boundary >> p.steps >> n)) die("malformed header");
    p.rule = parse_rule(rule);
    if (boundary != "torus" && boundary != "dead") die("boundary must be torus or dead");
    p.boundary = boundary == "torus" ? Boundary::Torus : Boundary::Dead;
    if (2 * p.rule.radius + 1 > std::min(p.width, p.height) || std::max(p.width, p.height) > MAX_SIDE ||
        p.steps < 0 || p.steps > MAX_STEPS)
        die("W, H or T out of range");

    auto inside = [&](int x, int y, int w, int h) {
        return x >= 0 && y >= 0 && w >= 0 && h >= 0 && x + w <= p.width && y + h <= p.height;
    };
    c.seeds.resize(n);
    for (Seed &s : c.seeds)
        if (!(in >> s.x >> s.y >> s.w >> s.h >> s.p >> s.seed) || !inside(s.x, s.y, s.w, s.h) || s.p < 0 || s.p > 100)
            die("malformed or out-of-range seed rectangle");
    return c;
}

// Calls f(i, y0, y1) for i in [0, row_threads(height)) on contiguous row ranges, a thread each.
static int row_threads(int height)
{
    return std::max(1, std::min<int>(std::thread::hardware_concurrency(), height));
}

template <class F>
static void for_rows(int height, F f)
{
    const int n = row_threads(height);
    std::vector<std::thread> ts;
    for (int i = 0; i < n; i++)
        ts.emplace_back(f, i, int(int64_t(height) * i / n), int(int64_t(height) * (i + 1) / n));
    for (auto &t : ts) t.join();
}

static void first_generation(uint8_t *grid, const Case &c)
{
    const int W = c.p.width;
    for_rows(c.p.height, [&](int, int y0, int y1) {
        std::fill(grid + size_t(y0) * W, grid + size_t(y1) * W, 0);
        for (const Seed &s : c.seeds)   // each rectangle's rows in [y0, y1), none of the others'
            for (int y = std::max(s.y, y0); y < std::min(s.y + s.h, y1); y++)
                for (int x = s.x; x < s.x + s.w; x++)
                    if (cell_hash(s.seed, x, y) % 1000000 < uint64_t(s.p) * 10000) grid[size_t(y) * W + x] = 1;
    });
}

// The hash is the sum, mod 2^64, of splitmix64((y * W + x) * C + state) over every cell not
// dead; a sum so the order cells are visited in cannot change it. pop[s] counts the cells in
// state s.
static void grid_hash(const uint8_t *grid, const Problem &p, uint64_t &hash, std::vector<uint64_t> &pop)
{
    const int W = p.width, C = p.rule.states, n = row_threads(p.height);
    std::vector<uint64_t> hashes(n, 0), pops(size_t(n) * C, 0);
    for_rows(p.height, [&](int i, int y0, int y1) {
        for (size_t k = size_t(y0) * W; k < size_t(y1) * W; k++) {
            const uint8_t s = grid[k];
            pops[size_t(i) * C + s]++;
            if (s) hashes[i] += splitmix64(k * C + s);
        }
    });
    hash = 0;
    pop.assign(C, 0);
    for (int i = 0; i < n; i++) {
        hash += hashes[i];
        for (int s = 0; s < C; s++) pop[s] += pops[size_t(i) * C + s];
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) die(std::string("usage: ") + argv[0] + " <input.txt> <output.txt>");

    /////////////////////////////////////////////////////////////
    // Input section. Outside the clock: reading the case, creating the CUDA context and building
    // the first generation in pinned host memory.
    const Case c = read_case(argv[1]);
    const Problem &p = c.p;
    const size_t cells = size_t(p.width) * p.height;
    auto l0 = std::chrono::steady_clock::now();
    cuda_check(cudaFree(nullptr), "creating the CUDA context");
    uint8_t *grid;
    cuda_check(cudaMallocHost(&grid, cells), "cudaMallocHost for the grid");
    first_generation(grid, c);
    /////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////
    // Timed section. The clock: run() and nothing else, until the device is idle.
    auto t0 = std::chrono::steady_clock::now();
    run(grid, p);
    cuda_check(cudaDeviceSynchronize(), "after run()");
    auto t1 = std::chrono::steady_clock::now();
    cuda_check(cudaGetLastError(), "after run()");
    /////////////////////////////////////////////////////////////

    for (size_t i = 0; i < cells; i++)
        if (grid[i] >= p.rule.states)
            die("run() left state " + std::to_string(grid[i]) + " at cell " + std::to_string(i) + ", outside [0, C)");

    /////////////////////////////////////////////////////////////
    // Output section. The judge parses this file.
    //   <hash, 16 hex digits>
    //   <cells in state 0> <cells in state 1> ... <cells in state C-1>
    uint64_t hash;
    std::vector<uint64_t> pop;
    grid_hash(grid, p, hash, pop);
    FILE *f = std::fopen(argv[2], "w");
    if (!f) die(std::string("cannot write ") + argv[2]);
    std::fprintf(f, "%016llx\n", (unsigned long long)hash);
    for (int s = 0; s < p.rule.states; s++)
        std::fprintf(f, "%llu%c", (unsigned long long)pop[s], s + 1 < p.rule.states ? ' ' : '\n');
    if (std::fclose(f) != 0) die(std::string("cannot write ") + argv[2]);
    /////////////////////////////////////////////////////////////

    cuda_check(cudaFreeHost(grid), "cudaFreeHost");
    std::cout << "Compute time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    std::cout << "Setup time: " << std::chrono::duration<double>(t0 - l0).count() << " s\n";
    return 0;
}
