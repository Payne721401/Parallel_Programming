// HW 2-2: C = A B with MPI, split by the inner index k.
//
// B is the big operand (2.3-2.9 GB on the performance cases, against 25-35 MB of A) and it
// starts out split by k: rank r holds B's rows [K r / P, K (r + 1) / P). So B stays put and A
// goes to it: every entry a = A[i, k] is sent to the rank holding B's row k, which adds
// a * B[k, :] into its piece of C's row i. The pieces then go to the rank that owns row i (it
// owns A's row i) and are summed there. Everything is private memory; across the two nodes
// only A and the pieces travel (about 18 MB and at most 120 MB on the public performance
// cases, measured), never B.
//
//   scatter   alltoallv of A's entries, packed (i << 32 | k << 4 | a), to B's owners
//   group     a counting sort of what arrived, by row i
//   bound     row i's piece from one rank has at most min(N, its products) entries; these
//             bounds, summed over the ranks, cut C's rows into rounds whose pieces fit a fixed
//             budget, so the send and receive buffers stay bounded even when C is dense
//   per round:
//     compute   each row's piece in a dense accumulator (N <= 8192: 32 KB) with a bitmap of
//               the columns hit, read back in column order
//     exchange  alltoallv of the pieces, a header (i, count) then count (col, val), to row i's
//               owner; rows are computed in order and their owners only grow, so the pieces
//               come out grouped by destination
//     merge     a row with one piece copies it; with several, sums them in the accumulator
//
// Measurement exists only with -DHW22_PROF (tests/run_hw2-2.sh builds it as hw2-2-prof): per
// phase the slowest and fastest rank, the products per rank, and the bytes each exchange sends,
// in all and across nodes.
#include "spgemm.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using u32 = uint32_t;
using u64 = uint64_t;

#ifdef HW22_PROF
class Prof {
public:
    static constexpr bool on = true;

    explicit Prof(MPI_Comm comm) : comm(comm)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        // A node is named by the lowest rank on it.
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int lead;
        MPI_Allreduce(&rank, &lead, 1, MPI_INT, MPI_MIN, node);
        MPI_Comm_free(&node);
        node_of.resize(size);
        MPI_Allgather(&lead, 1, MPI_INT, node_of.data(), 1, MPI_INT, comm);
        last = MPI_Wtime();
    }

    // Closes the phase that has run since the previous mark; repeated names add up.
    void mark(const char *name)
    {
        const double t = MPI_Wtime();
        find(phases, name) += t - last;
        last = t;
    }

    void count(const char *name, u64 v) { find(counts, name) += v; }

    // The bytes this rank sends in an exchange, in all and to other nodes.
    void traffic(const char *name, const std::vector<u64> &items_to, u64 item_bytes)
    {
        u64 all = 0, away = 0;
        for (int q = 0; q < size; q++) {
            all += items_to[q] * item_bytes;
            if (node_of[q] != node_of[rank]) away += items_to[q] * item_bytes;
        }
        count((std::string(name) + "-bytes").c_str(), all);
        count((std::string(name) + "-xnode").c_str(), away);
    }

    void report()
    {
        for (auto &p : phases) {
            double mx = 0, mn = 0;
            MPI_Reduce(&p.second, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&p.second, &mn, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
            if (rank == 0) std::fprintf(stderr, "[hw2-2] %-14s max %9.4f s  min %9.4f s\n", p.first.c_str(), mx, mn);
        }
        for (auto &c : counts) {
            u64 sum = 0, mx = 0;
            MPI_Reduce(&c.second, &sum, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
            MPI_Reduce(&c.second, &mx, 1, MPI_UINT64_T, MPI_MAX, 0, comm);
            if (rank == 0)
                std::fprintf(stderr, "[hw2-2] %-14s sum %llu  max/rank %llu  max/mean %.2f\n", c.first.c_str(),
                             (unsigned long long)sum, (unsigned long long)mx,
                             sum ? double(mx) * size / double(sum) : 0.0);
        }
    }

private:
    MPI_Comm comm;
    int rank = 0, size = 1;
    std::vector<int> node_of;
    std::vector<std::pair<std::string, double>> phases;
    std::vector<std::pair<std::string, u64>> counts;
    double last = 0;

    template <class T>
    static T &find(std::vector<std::pair<std::string, T>> &v, const std::string &name)
    {
        for (auto &p : v)
            if (p.first == name) return p.second;
        v.push_back({name, T()});
        return v.back().second;
    }
};
#else
// The judge's build: measurement compiled out, every call an empty inline.
class Prof {
public:
    static constexpr bool on = false;
    explicit Prof(MPI_Comm) {}
    void mark(const char *) {}
    void count(const char *, u64) {}
    void traffic(const char *, const std::vector<u64> &, u64) {}
    void report() {}
};
#endif

int count_of(u64 n)
{
    if (n > u64(INT_MAX)) throw std::runtime_error("an MPI count exceeds INT_MAX");
    return int(n);
}

// Alltoallv of 8-byte items: send holds to[q] items for each rank q, in rank order. Returns
// what arrived, in source order; from[q] is how many came from q.
template <class T>
std::vector<T> alltoallv(const std::vector<T> &send, const std::vector<u64> &to, std::vector<u64> &from,
                         MPI_Comm comm)
{
    static_assert(sizeof(T) == 8, "items travel as MPI_UINT64_T");
    const int P = int(to.size());
    from.assign(P, 0);
    MPI_Alltoall(to.data(), 1, MPI_UINT64_T, from.data(), 1, MPI_UINT64_T, comm);
    std::vector<int> sc(P), sd(P), rc(P), rd(P);
    u64 so = 0, ro = 0;
    for (int q = 0; q < P; q++) {
        sc[q] = count_of(to[q]);
        sd[q] = count_of(so);
        so += to[q];
        rc[q] = count_of(from[q]);
        rd[q] = count_of(ro);
        ro += from[q];
    }
    count_of(so);
    count_of(ro);
    std::vector<T> recv(ro);
    MPI_Alltoallv(send.data(), sc.data(), sd.data(), MPI_UINT64_T, recv.data(), rc.data(), rd.data(),
                  MPI_UINT64_T, comm);
    return recv;
}

// One row of C being summed: a dense total per column (they fit 32 bits: C[i, j] is at most
// 64 nnz(A[i, :]) <= 2^23), a bitmap of the columns hit, and the span they lie in, so that
// reading the row back in column order costs the span's words rather than N.
class Accumulator {
public:
    explicit Accumulator(u32 n) : sum(n, 0), hit((u64(n) + 63) / 64, 0) {}

    // Adds scale times a row whose columns ascend. The bits reach the bitmap a word at a time:
    // a run of columns shares a word, and or-ing each one into memory would chain every store
    // to the next load (measured 3.6 times slower on a dense row).
    void add(u32 scale, const Entry *e, u64 len)
    {
        if (!len) return;
        lo = std::min(lo, e[0].col);
        hi = std::max(hi, e[len - 1].col);
        u32 w = e[0].col / 64;
        u64 bits = 0;
        for (u64 t = 0; t < len; t++) {
            const u32 j = e[t].col;
            sum[j] += scale * e[t].val;
            if (j / 64 != w) {
                hit[w] |= bits;
                w = j / 64;
                bits = 0;
            }
            bits |= u64(1) << (j % 64);
        }
        hit[w] |= bits;
    }

    // Appends the row to out in column order, and clears it. Every product is positive, so
    // every column hit has a nonzero total.
    void take(std::vector<Entry> &out)
    {
        if (lo > hi) return;
        for (u32 w = lo / 64; w <= hi / 64; w++) {
            u64 x = hit[w];
            hit[w] = 0;
            while (x) {
                const u32 j = w * 64 + u32(__builtin_ctzll(x));
                out.push_back({j, sum[j]});
                sum[j] = 0;
                x &= x - 1;
            }
        }
        lo = UINT32_MAX;
        hi = 0;
    }

private:
    std::vector<u32> sum;
    std::vector<u64> hit;
    u32 lo = UINT32_MAX, hi = 0;
};

// What one round may put in flight on any rank, either way: 8-byte items, 128 MiB.
constexpr u64 BUDGET = u64(16) << 20;

}  // namespace

void multiply(const Matrix &a, const Matrix &b, Matrix &c)
{
    const MPI_Comm comm = MPI_COMM_WORLD;
    int P, r;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &r);
    Prof prof(comm);
    const u32 M = a.rows, K = a.cols, N = b.cols;

    // A's entries to the owners of their B rows.
    std::vector<u64> to(P, 0), from;
    for (const Entry &x : a.data) to[row_owner(x.col, K, P)]++;
    std::vector<u64> at(P, 0);
    for (int q = 1; q < P; q++) at[q] = at[q - 1] + to[q - 1];
    std::vector<u64> send(a.data.size());
    for (u32 il = 0; il < a.local_rows(); il++) {
        const u64 i = a.first + il;
        for (u64 e = a.ptr[il]; e < a.ptr[il + 1]; e++) {
            const Entry x = a.data[e];
            send[at[row_owner(x.col, K, P)]++] = i << 32 | u64(x.col) << 4 | x.val;
        }
    }
    prof.traffic("a", to, 8);
    std::vector<u64> got = alltoallv(send, to, from, comm);
    std::vector<u64>().swap(send);
    prof.mark("scatter");

    // Grouped by row: row i's (k << 4 | a) are item[first[i], first[i + 1]).
    std::vector<u64> first(u64(M) + 1, 0);
    for (const u64 x : got) first[(x >> 32) + 1]++;
    for (u32 i = 0; i < M; i++) first[i + 1] += first[i];
    std::vector<u32> item(got.size());
    {
        std::vector<u64> cur(first.begin(), first.end() - 1);
        for (const u64 x : got) item[cur[x >> 32]++] = u32(x);
    }
    std::vector<u64>().swap(got);
    prof.mark("group");

    // Row i's piece from this rank, and over all ranks (a header each included).
    std::vector<u64> mine(M, 0), bound(M, 0);
    u64 products = 0;
    for (u32 i = 0; i < M; i++) {
        u64 f = 0;
        for (u64 t = first[i]; t < first[i + 1]; t++) {
            const u32 kl = (item[t] >> 4) - b.first;
            f += b.ptr[kl + 1] - b.ptr[kl];
        }
        products += f;
        mine[i] = f ? std::min<u64>(N, f) + 1 : 0;
    }
    MPI_Allreduce(mine.data(), bound.data(), count_of(M), MPI_UINT64_T, MPI_SUM, comm);
    prof.count("products", products);
    prof.mark("bound");

    c.rows = M;
    c.cols = N;
    c.first = a.first;
    c.ptr.assign(u64(a.local_rows()) + 1, 0);
    c.data.clear();
    const u32 my0 = a.first, my1 = a.first + a.local_rows();
    Accumulator acc(N);
    std::vector<Entry> out;
    std::vector<u64> to_c(P), from_c, at_c(P), end_c(P);
    u64 rounds = 0;
    for (u32 lo = 0; lo < M;) {
        // This round's rows: as many as fit the budget, and at least one.
        u32 hi = lo;
        u64 load = 0, mine_load = 0;
        while (hi < M && (hi == lo || load + bound[hi] <= BUDGET)) {
            load += bound[hi];
            mine_load += mine[hi];
            hi++;
        }

        out.clear();
        out.reserve(mine_load);
        std::fill(to_c.begin(), to_c.end(), 0);
        for (u32 i = lo; i < hi; i++) {
            if (first[i] == first[i + 1]) continue;
            for (u64 t = first[i]; t < first[i + 1]; t++) {
                const u32 kl = (item[t] >> 4) - b.first;
                acc.add(item[t] & 15, b.data.data() + b.ptr[kl], b.ptr[kl + 1] - b.ptr[kl]);
            }
            const u64 head = out.size();
            out.push_back({i, 0});
            acc.take(out);
            if (out.size() == head + 1) {   // nothing: the B rows it met were empty
                out.pop_back();
                continue;
            }
            out[head].val = u32(out.size() - head - 1);
            to_c[row_owner(i, M, P)] += out.size() - head;
        }
        prof.mark("compute");
        prof.traffic("c", to_c, 8);
        const std::vector<Entry> in = alltoallv(out, to_c, from_c, comm);
        prof.mark("exchange");

        // My rows of the round, in order; each source's pieces come in row order.
        u64 o = 0;
        for (int q = 0; q < P; q++) {
            at_c[q] = o;
            o += from_c[q];
            end_c[q] = o;
        }
        for (u32 i = std::max(lo, my0); i < std::min(hi, my1); i++) {
            const Entry *one = nullptr;
            u32 one_len = 0;
            int pieces = 0;
            for (int q = 0; q < P; q++) {
                if (at_c[q] == end_c[q] || in[at_c[q]].col != i) continue;
                const u32 len = in[at_c[q]].val;
                const Entry *piece = in.data() + at_c[q] + 1;
                at_c[q] += 1 + u64(len);
                if (++pieces == 1) {
                    one = piece;
                    one_len = len;
                    continue;
                }
                if (pieces == 2) acc.add(1, one, one_len);
                acc.add(1, piece, len);
            }
            if (pieces == 1)
                c.data.insert(c.data.end(), one, one + one_len);
            else if (pieces > 1)
                acc.take(c.data);
            c.ptr[i - my0 + 1] = c.data.size();
        }
        prof.mark("merge");
        rounds++;
        lo = hi;
    }
    prof.count("rounds", r == 0 ? rounds : 0);
    prof.report();
}
