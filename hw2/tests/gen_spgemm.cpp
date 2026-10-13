// Writes an SpGEMM test case, A.csr and B.csr in SPEC.md's binary CSR layout, into a directory;
// the starter (hw2-2-seq) then computes its C.csr. Kinds, for the boundaries SPEC.md lists that
// no public case reaches:
//
//   dense  <dir>                 M 1024, K 525, N 8192, A and B full: C is complete (every
//                                entry stored) and the scalar products total 4.40e9 > 2^32
//   nnza   <dir>                 M 8192, K 4194304 (the maximum), N 8192: 1024 entries in every
//                                row of A, nnz(A) = 8388608 (the maximum); B rows hold 0-2
//   random <dir> <seed> <M> <K> <N>   small random shapes: empty rows, rows of every density,
//                                values 1..8, columns that cluster
//
// Run it through srun; it holds both matrices in memory (about 100 MB for nnza).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using u32 = uint32_t;
using u64 = uint64_t;
struct Entry {
    u32 col, val;
};
using Rows = std::vector<std::vector<Entry>>;

void write(const std::string &path, u32 rows, u32 cols, const Rows &r)
{
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::perror(path.c_str());
        std::exit(1);
    }
    u64 nnz = 0;
    for (const auto &x : r) nnz += x.size();
    std::fwrite("SPGCSR01", 1, 8, f);
    std::fwrite(&rows, 4, 1, f);
    std::fwrite(&cols, 4, 1, f);
    std::fwrite(&nnz, 8, 1, f);
    u64 o = 0;
    std::fwrite(&o, 8, 1, f);
    for (const auto &x : r) {
        o += x.size();
        std::fwrite(&o, 8, 1, f);
    }
    for (const auto &x : r) std::fwrite(x.data(), sizeof(Entry), x.size(), f);
    std::fclose(f);
}

// n distinct columns of [0, cols), ascending, values 1..8.
std::vector<Entry> pick(std::mt19937_64 &g, u32 cols, u32 n)
{
    std::vector<u32> c;
    if (u64(n) * 2 >= cols) {   // dense: keep each column with probability n / cols
        for (u32 j = 0; j < cols; j++)
            if (g() % cols < n) c.push_back(j);
    } else {
        while (c.size() < n) {
            while (c.size() < n) c.push_back(u32(g() % cols));
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
        }
    }
    std::vector<Entry> r;
    r.reserve(c.size());
    for (const u32 j : c) r.push_back({j, u32(1 + g() % 8)});
    return r;
}

std::vector<Entry> full(std::mt19937_64 &g, u32 cols)
{
    std::vector<Entry> r(cols);
    for (u32 j = 0; j < cols; j++) r[j] = {j, u32(1 + g() % 8)};
    return r;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s dense|nnza <dir> | random <dir> <seed> <M> <K> <N>\n", argv[0]);
        return 2;
    }
    const std::string kind = argv[1], dir = argv[2];
    std::mt19937_64 g(2026);
    Rows A, B;
    u32 M, K, N;
    if (kind == "dense") {
        M = 1024, K = 525, N = 8192;
        A.resize(M);
        B.resize(K);
        for (auto &x : A) x = full(g, K);
        for (auto &x : B) x = full(g, N);
    } else if (kind == "nnza") {
        M = 8192, K = 4194304, N = 8192;
        A.resize(M);
        B.resize(K);
        for (auto &x : A) x = pick(g, K, 1024);
        for (auto &x : B) x = pick(g, N, u32(g() % 3));
    } else if (kind == "random" && argc == 7) {
        g.seed(std::strtoull(argv[3], nullptr, 10));
        M = u32(std::atoi(argv[4])), K = u32(std::atoi(argv[5])), N = u32(std::atoi(argv[6]));
        A.resize(M);
        B.resize(K);
        // Each row is empty, sparse, clustered at low columns, or dense, at random.
        auto row = [&](u32 cols) {
            switch (g() % 4) {
            case 0: return std::vector<Entry>();
            case 1: return pick(g, cols, u32(1 + g() % std::max<u32>(1, cols / 16)));
            case 2: return pick(g, std::max<u32>(1, cols / 8), u32(1 + g() % std::max<u32>(1, cols / 16)));
            default: return pick(g, cols, u32(1 + g() % cols));
            }
        };
        for (auto &x : A) x = row(K);
        for (auto &x : B) x = row(N);
    } else {
        std::fprintf(stderr, "unknown kind or wrong arguments: %s\n", kind.c_str());
        return 2;
    }
    write(dir + "/A.csr", M, K, A);
    write(dir + "/B.csr", K, N, B);
    u64 na = 0, nb = 0, flops = 0;
    std::vector<u64> blen(K);
    for (u32 k = 0; k < K; k++) nb += blen[k] = B[k].size();
    for (const auto &x : A) {
        na += x.size();
        for (const Entry &e : x) flops += blen[e.col];
    }
    std::printf("%s: M %u K %u N %u  nnz A %llu  nnz B %llu  products %llu\n", kind.c_str(), M, K, N,
                (unsigned long long)na, (unsigned long long)nb, (unsigned long long)flops);
    return 0;
}
