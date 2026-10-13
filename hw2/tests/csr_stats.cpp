// HW 2-2 case statistics, for choosing how to distribute the work. Serial: on SEIREN run it
// through srun (srun -N1 -n1 ./csr_stats A.csr B.csr C.csr [P ...]), never on the login node.
//
// For one case it prints:
//   sizes      M, K, N, nnz of A, B, C, and the scalar products ("flops", sum over A[i,k] of
//              nnz(B[k,:])) against nnz(C);
//   rows       how the flops spread over A rows and B rows (max, percentiles, top shares);
//   per P      for each rank count P (two nodes, ranks split in halves as --ntasks-per-node
//              places them): the flops per rank if each rank computes its own A rows (row-wise)
//              or the products of its own B rows (K-split), per rank and per node; the bytes
//              crossing nodes if each node pulls the B rows its rows need, or pushes its A
//              entries to the B rows' node; and partial-C sizes, as upper bounds
//              (min(N, flops) per row and piece) and, on sample rows, exactly;
//   kernel     the course's LocalAccumulator and a dense uint32 accumulator with a bitmap, timed
//              on real rows (ns per scalar product), checked against C's row lengths.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "local_kernel.hpp"

namespace {

using u32 = uint32_t;
using u64 = uint64_t;

struct Csr {
    u32 rows = 0, cols = 0;
    u64 nnz = 0;
    std::vector<u64> ptr;
    std::vector<Entry> e;
};

Csr load(const char *path, bool entries)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    char magic[8];
    Csr m;
    f.read(magic, 8);
    f.read(reinterpret_cast<char *>(&m.rows), 4);
    f.read(reinterpret_cast<char *>(&m.cols), 4);
    f.read(reinterpret_cast<char *>(&m.nnz), 8);
    if (!f || std::memcmp(magic, "SPGCSR01", 8) != 0) throw std::runtime_error(std::string("bad header in ") + path);
    m.ptr.resize(u64(m.rows) + 1);
    f.read(reinterpret_cast<char *>(m.ptr.data()), std::streamsize(8 * m.ptr.size()));
    if (entries) {
        m.e.resize(m.nnz);
        f.read(reinterpret_cast<char *>(m.e.data()), std::streamsize(8 * m.nnz));
    }
    if (!f) throw std::runtime_error(std::string("short file ") + path);
    return m;
}

double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Value at fraction q of the sorted copy of v.
template <class T>
T quantile(std::vector<T> v, double q)
{
    if (v.empty()) return T(0);
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(q * double(v.size())))];
}

// Share of the total held by the largest fraction f of the values.
template <class T>
double top_share(std::vector<T> v, double f)
{
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end(), std::greater<T>());
    const size_t n = std::max<size_t>(1, size_t(f * double(v.size())));
    double top = 0, all = 0;
    for (size_t i = 0; i < v.size(); i++) {
        all += double(v[i]);
        if (i < n) top += double(v[i]);
    }
    return all > 0 ? top / all : 0;
}

const char *gb(double bytes)
{
    static char buf[8][32];
    static int k = 0;
    char *s = buf[k++ & 7];
    if (bytes >= 1e9)
        std::snprintf(s, 32, "%.2f GB", bytes / 1e9);
    else
        std::snprintf(s, 32, "%.1f MB", bytes / 1e6);
    return s;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s A.csr B.csr C.csr [P ...]\n", argv[0]);
        return 2;
    }
    std::vector<int> Ps;
    for (int i = 4; i < argc; i++) Ps.push_back(std::atoi(argv[i]));
    if (Ps.empty()) Ps = {8, 24};
    constexpr int NODES = 2;

    const double t0 = now();
    const Csr A = load(argv[1], true), B = load(argv[2], true), C = load(argv[3], false);
    const u32 M = A.rows, K = A.cols, N = B.cols;
    std::printf("== sizes  (loaded in %.1f s)\n", now() - t0);
    std::printf("M %u  K %u  N %u   nnz A %llu  B %llu  C %llu   CSR bytes A %s  B %s  C %s\n", M, K, N,
                (unsigned long long)A.nnz, (unsigned long long)B.nnz, (unsigned long long)C.nnz,
                gb(8.0 * A.nnz + 8.0 * (M + 1)), gb(8.0 * B.nnz + 8.0 * (K + 1)), gb(8.0 * C.nnz + 8.0 * (M + 1)));

    // b[k]: length of B row k; a[k]: entries of A in column k; per A row: entries and flops.
    std::vector<u32> b(K), a(K, 0);
    for (u32 k = 0; k < K; k++) b[k] = u32(B.ptr[k + 1] - B.ptr[k]);
    std::vector<u64> rflops(M, 0), rnnz(M), cnnz(M);
    u64 flops = 0;
    for (u32 i = 0; i < M; i++) {
        rnnz[i] = A.ptr[i + 1] - A.ptr[i];
        cnnz[i] = C.ptr[i + 1] - C.ptr[i];
        for (u64 q = A.ptr[i]; q < A.ptr[i + 1]; q++) {
            const u32 k = A.e[q].col;
            a[k]++;
            rflops[i] += b[k];
        }
        flops += rflops[i];
    }
    u64 used = 0, bref = 0, bmax = 0, amax = 0;
    std::vector<double> kwork;   // a_k * b_k over the k that A uses
    for (u32 k = 0; k < K; k++) {
        bmax = std::max<u64>(bmax, b[k]);
        if (!a[k]) continue;
        used++;
        bref += b[k];
        amax = std::max<u64>(amax, a[k]);
        kwork.push_back(double(a[k]) * b[k]);
    }
    std::printf("flops %llu (%.2fe9)   flops / nnz(C) %.1f   B rows used by A %llu of %u (%.1f%%), holding %.1f%% of B\n",
                (unsigned long long)flops, flops / 1e9, C.nnz ? double(flops) / double(C.nnz) : 0.0,
                (unsigned long long)used, K, 100.0 * used / K, B.nnz ? 100.0 * bref / B.nnz : 0.0);

    std::printf("== rows\n");
    u64 nonempty = 0;
    for (u32 i = 0; i < M; i++) nonempty += rnnz[i] > 0;
    std::printf("A rows nonempty %llu of %u; nnz per A row: max %llu  p50 %llu  p99 %llu\n", (unsigned long long)nonempty,
                M, (unsigned long long)quantile(rnnz, 1.0), (unsigned long long)quantile(rnnz, 0.5),
                (unsigned long long)quantile(rnnz, 0.99));
    std::printf("flops per A row: max %llu  p50 %llu  p99 %llu  mean %.0f;  top 1%% of rows %.1f%%, top 10%% %.1f%% of flops\n",
                (unsigned long long)quantile(rflops, 1.0), (unsigned long long)quantile(rflops, 0.5),
                (unsigned long long)quantile(rflops, 0.99), double(flops) / M, 100 * top_share(rflops, 0.01),
                100 * top_share(rflops, 0.10));
    std::printf("nnz per C row: max %llu  p50 %llu  p99 %llu  (N = %u)\n", (unsigned long long)quantile(cnnz, 1.0),
                (unsigned long long)quantile(cnnz, 0.5), (unsigned long long)quantile(cnnz, 0.99), N);
    std::printf("B row length: max %llu  mean over used rows %.1f;  A column count a_k: max %llu  mean over used %.2f\n",
                (unsigned long long)bmax, used ? double(bref) / used : 0.0, (unsigned long long)amax,
                used ? double(A.nnz) / used : 0.0);
    std::printf("work a_k*b_k per used B row: top 0.1%% of rows %.1f%%, top 1%% %.1f%% of flops\n",
                100 * top_share(kwork, 0.001), 100 * top_share(kwork, 0.01));
    // Used B rows from the most work (a_k * b_k) down, and each row's place in that order.
    std::vector<u32> hot_order;
    for (u32 k = 0; k < K; k++)
        if (a[k]) hot_order.push_back(k);
    std::sort(hot_order.begin(), hot_order.end(),
              [&](u32 x, u32 y) { return double(a[x]) * b[x] > double(a[y]) * b[y]; });
    std::vector<u32> hot_rank(K, ~0u);
    for (size_t j = 0; j < hot_order.size(); j++) hot_rank[hot_order[j]] = u32(j);

    for (const int P : Ps) {
        auto owner = [&](u64 n, u64 x) { return int(((x + 1) * u64(P) - 1) / n); };
        auto node = [&](int r) { return r * NODES / P; };
        std::vector<double> roww(P, 0), kw(P, 0), abytes(P, 0), bbytes(P, 0);
        for (u32 i = 0; i < M; i++) {
            roww[owner(M, i)] += double(rflops[i]);
            abytes[owner(M, i)] += 8.0 * rnnz[i];
        }
        for (u32 k = 0; k < K; k++) {
            kw[owner(K, k)] += double(a[k]) * b[k];
            bbytes[owner(K, k)] += 8.0 * b[k];
        }
        auto summary = [&](const char *what, const std::vector<double> &w) {
            double mx = 0, sum = 0, nd[NODES] = {};
            for (int r = 0; r < P; r++) {
                mx = std::max(mx, w[r]);
                sum += w[r];
                nd[node(r)] += w[r];
            }
            std::printf("  %-28s max/mean %.2f   node0 %.1f%%  node1 %.1f%%\n", what, sum > 0 ? mx / (sum / P) : 0.0,
                        sum > 0 ? 100 * nd[0] / sum : 0.0, sum > 0 ? 100 * nd[1] / sum : 0.0);
        };
        std::printf("== P = %d (%d nodes, %d ranks each)\n", P, NODES, P / NODES);
        summary("row-wise flops per rank", roww);
        summary("K-split flops per rank", kw);
        // K-split with the heaviest B rows replicated: their work can go to any rank, the rest
        // stays with B's owners. The hot work fills ranks up to the mean, so the max is the mean
        // unless some rank's own cold work is above it.
        for (const double f : {0.001, 0.01, 0.05}) {
            const u64 n = std::max<u64>(1, u64(f * double(hot_order.size())));
            std::vector<double> cold(P, 0);
            double hot = 0, bytes = 0, nd[NODES] = {};
            for (u32 k = 0; k < K; k++) {
                if (!a[k]) continue;
                const double w = double(a[k]) * b[k];
                if (hot_rank[k] < n) {
                    hot += w;
                    bytes += 8.0 * b[k];
                } else {
                    cold[owner(K, k)] += w;
                    nd[node(owner(K, k))] += w;
                }
            }
            const double total = hot + nd[0] + nd[1], mean = total / P;
            const double mx = std::max(mean, *std::max_element(cold.begin(), cold.end()));
            std::printf("  K-split, top %4.1f%% B rows (%7llu rows, %s) replicated: hot work %.1f%%, max/mean %.2f,"
                        " cold work node0 %.1f%% node1 %.1f%%\n",
                        100 * f, (unsigned long long)n, gb(bytes), total > 0 ? 100 * hot / total : 0.0,
                        mean > 0 ? mx / mean : 0.0, total > 0 ? 100 * nd[0] / total : 0.0,
                        total > 0 ? 100 * nd[1] / total : 0.0);
        }
        std::printf("  initial shards per rank: A max %s, B max %s\n", gb(*std::max_element(abytes.begin(), abytes.end())),
                    gb(*std::max_element(bbytes.begin(), bbytes.end())));

        // Cross-node traffic. Pull: a node fetches the B rows (on the other node) its rows use.
        // Push: a node sends the A entries whose B row lives on the other node, and gets back
        // partial rows (bounded by min(N, flops of the pushed entries) per row).
        // K-split everywhere: every rank multiplies its own B rows; per A row, each rank holding
        // some of its k's makes one partial row (bounded by min(N, flops of that piece)).
        std::vector<std::vector<uint8_t>> need(NODES, std::vector<uint8_t>(K, 0));
        double push[NODES] = {}, ret[NODES] = {}, kpart = 0, kpart_cross = 0, pieces = 0;
        for (u32 i = 0; i < M; i++) {
            const int ni = node(owner(M, i));
            u64 cross = 0, piece = 0;
            int pr = -1;
            for (u64 q = A.ptr[i]; q <= A.ptr[i + 1]; q++) {
                const int r = q < A.ptr[i + 1] ? owner(K, A.e[q].col) : -2;
                if (r != pr && pr >= 0) {   // close the piece for rank pr
                    const double s = double(std::min<u64>(N, piece));
                    kpart += s;
                    if (node(pr) != ni) kpart_cross += s;
                    pieces++;
                    piece = 0;
                }
                if (r < 0) break;
                pr = r;
                const u32 k = A.e[q].col;
                piece += b[k];
                if (node(r) != ni) {
                    push[ni] += 8;
                    cross += b[k];
                    need[ni][k] = 1;
                }
            }
            ret[ni] += 8.0 * std::min<u64>(N, cross);
        }
        double pull[NODES] = {};
        for (int n = 0; n < NODES; n++)
            for (u32 k = 0; k < K; k++)
                if (need[n][k]) pull[n] += 8.0 * b[k];
        std::printf("  pull B rows across nodes:     into node0 %s, into node1 %s\n", gb(pull[0]), gb(pull[1]));
        std::printf("  push A entries across nodes:  from node0 %s, from node1 %s; partial rows back <= %s, %s\n",
                    gb(push[0]), gb(push[1]), gb(ret[0]), gb(ret[1]));
        std::printf("  K-split partial C (all ranks): <= %s in %.0f pieces (%.1f per nonempty row), crossing nodes <= %s\n",
                    gb(8 * kpart), pieces, nonempty ? pieces / nonempty : 0.0, gb(8 * kpart_cross));
    }

    // Exact partial sizes on sample rows: the 8 heaviest and 8 spread over the rest, for the last
    // P. A stamp per column counts each distinct column once per piece.
    const int P = Ps.back();
    auto owner = [&](u64 n, u64 x) { return int(((x + 1) * u64(P) - 1) / n); };
    std::vector<u32> order;
    for (u32 i = 0; i < M; i++)
        if (rnnz[i]) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](u32 x, u32 y) { return rflops[x] > rflops[y]; });
    std::vector<u32> sample(order.begin(), order.begin() + std::min<size_t>(8, order.size()));
    for (size_t j = 1; j <= 8 && order.size() > 8; j++) sample.push_back(order[8 + (order.size() - 9) * j / 8]);
    std::printf("== sample rows, P = %d: exact partial sizes vs the min(N, flops) bound\n", P);
    std::vector<u32> stamp(N, 0);
    u32 tick = 0;
    for (const u32 i : sample) {
        u64 full = 0, parts = 0, bound = 0, npieces = 0;
        ++tick;
        for (u64 q = A.ptr[i]; q < A.ptr[i + 1]; q++)
            for (u64 j = B.ptr[A.e[q].col]; j < B.ptr[A.e[q].col + 1]; j++)
                if (stamp[B.e[j].col] != tick) {
                    stamp[B.e[j].col] = tick;
                    full++;
                }
        int pr = -1;
        u64 piece = 0;
        for (u64 q = A.ptr[i]; q < A.ptr[i + 1]; q++) {
            const u32 k = A.e[q].col;
            const int r = owner(K, k);
            if (r != pr) {
                if (pr >= 0) bound += std::min<u64>(N, piece);
                piece = 0;
                pr = r;
                npieces++;
                ++tick;
            }
            piece += b[k];
            for (u64 j = B.ptr[k]; j < B.ptr[k + 1]; j++)
                if (stamp[B.e[j].col] != tick) {
                    stamp[B.e[j].col] = tick;
                    parts++;
                }
        }
        if (pr >= 0) bound += std::min<u64>(N, piece);
        std::printf("  row %5u: nnz A %6llu flops %11llu  nnz C %5llu (exact %5llu %s)  pieces %2llu  partial exact %8llu"
                    " bound %8llu\n",
                    i, (unsigned long long)rnnz[i], (unsigned long long)rflops[i], (unsigned long long)cnnz[i],
                    (unsigned long long)full, full == cnnz[i] ? "ok" : "MISMATCH", (unsigned long long)npieces,
                    (unsigned long long)parts, (unsigned long long)bound);
    }

    // Kernel speed on real rows: the course's accumulator (uint64 values, touched list, sort)
    // against a dense uint32 accumulator whose bitmap yields the columns already sorted.
    std::printf("== kernel, one core: ns per scalar product\n");
    LocalAccumulator course(N);
    std::vector<u32> acc(N, 0);
    std::vector<u64> bits((N + 63) / 64, 0);
    double tc = 0, td = 0, fl = 0;
    u64 rows = 0, mismatch = 0;
    for (const u32 i : order) {   // heaviest first, until about 2e9 products
        if (fl > 2e9) break;
        if (rflops[i] > 1e9) continue;
        double s = now();
        for (u64 q = A.ptr[i]; q < A.ptr[i + 1]; q++) {
            const u32 k = A.e[q].col;
            if (b[k]) course.add_scaled_row(A.e[q].val, B.e.data() + B.ptr[k], b[k]);
        }
        const u64 nc = course.finish().size();
        tc += now() - s;
        s = now();
        for (u64 q = A.ptr[i]; q < A.ptr[i + 1]; q++) {
            const u32 k = A.e[q].col, av = A.e[q].val;
            for (u64 j = B.ptr[k]; j < B.ptr[k + 1]; j++) {
                const u32 c = B.e[j].col;
                acc[c] += av * B.e[j].val;
                bits[c >> 6] |= u64(1) << (c & 63);
            }
        }
        u64 nd = 0;
        for (size_t w = 0; w < bits.size(); w++)
            for (u64 m = bits[w]; m; m &= m - 1) {
                const u32 c = u32(w * 64 + __builtin_ctzll(m));
                nd += acc[c] != 0;
                acc[c] = 0;
            }
        std::fill(bits.begin(), bits.end(), 0);
        td += now() - s;
        mismatch += nc != cnnz[i] || nd != cnnz[i];
        fl += double(rflops[i]);
        rows++;
    }
    std::printf("  %llu rows, %.2fe9 products: LocalAccumulator %.2f ns, dense+bitmap %.2f ns;  row lengths vs C: %s\n",
                (unsigned long long)rows, fl / 1e9, fl > 0 ? tc * 1e9 / fl : 0.0, fl > 0 ? td * 1e9 / fl : 0.0,
                mismatch ? "MISMATCH" : "ok");
    return 0;
}
