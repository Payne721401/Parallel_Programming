// HW 2-2: C = A B with MPI, split by the inner index k.
//
// B is the big operand (2.3-2.9 GB on the performance cases, against 25-35 MB of A) and it
// starts out split by k: rank r holds B's rows [K r / P, K (r + 1) / P). So B stays put and A
// goes to it: every entry a = A[i, k] is sent to the rank holding B's row k, which adds
// a * B[k, :] into its piece of C's row i. The pieces then go to the rank that owns row i (it
// owns A's row i) and are summed there. This remains the default and the cold-row path.
// PP_HOT=1 optionally shares selected B rows once per node and assigns each output row's
// hot work to one worker, weighted by products and the workers' remaining cold work.
// PP_HOT_MB bounds that cache per node (default 256, range 0..512). PP_HOT_REPORT=1 prints
// the selected plan even in a normal build; profiling builds report it by default.
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
#include <cstdlib>
#include <exception>
#include <numeric>
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

struct HotConfig {
    bool enabled, report;
    u64 bytes;
};

HotConfig hot_config(MPI_Comm comm, int rank)
{
    int opt[3] = {0, 256, Prof::on ? 1 : 0};
    if (rank == 0) {
        const char *names[] = {"PP_HOT", "PP_HOT_MB", "PP_HOT_REPORT"};
        for (int i = 0; i < 3; i++) {
            const char *s = std::getenv(names[i]);
            if (!s) continue;
            char *end = nullptr;
            const long v = std::strtol(s, &end, 10);
            if (!*s || *end || v < 0 || v > (i == 1 ? 512 : 1))
                throw std::runtime_error(std::string("invalid ") + names[i]);
            opt[i] = int(v);
        }
    }
    MPI_Bcast(opt, 3, MPI_INT, 0, comm);
    return {opt[0] != 0, opt[2] != 0, u64(opt[1]) << 20};
}

struct HotPlan {
    static constexpr u64 NONE = UINT64_MAX;
    std::vector<u32> length;
    std::vector<u64> offset, starts;
    std::vector<int> worker;
    u64 total = 0, before = 0, after = 0, rows = 0, entries = 0, hot_products = 0;

    bool contains(u32 k) const { return !offset.empty() && offset[k] != NONE; }
};

HotPlan plan_hot(const Matrix &a, const Matrix &b, const HotConfig &cfg, MPI_Comm comm, int P)
{
    HotPlan h;
    if (!cfg.enabled || !cfg.bytes || P == 1) return h;
    const u32 K = a.cols, M = a.rows;
    const u64 local_nnz[2] = {a.data.size(), b.data.size()};
    u64 global_nnz[2];
    MPI_Allreduce(local_nnz, global_nnz, 2, MPI_UINT64_T, MPI_SUM, comm);
    if (!global_nnz[0] || !global_nnz[1]) return h;
    std::vector<u32> uses(K, 0), lengths(K), local(b.local_rows());
    for (const Entry &x : a.data) uses[x.col]++;
    MPI_Allreduce(MPI_IN_PLACE, uses.data(), count_of(K), MPI_UINT32_T, MPI_SUM, comm);
    std::vector<int> counts(P), displs(P);
    for (int q = 0; q < P; q++) {
        displs[q] = count_of(block_start(K, q, P));
        counts[q] = count_of(block_start(K, q + 1, P) - u32(displs[q]));
    }
    for (u32 k = 0; k < b.local_rows(); k++) local[k] = u32(b.ptr[k + 1] - b.ptr[k]);
    MPI_Allgatherv(local.data(), count_of(local.size()), MPI_UINT32_T, lengths.data(),
                   counts.data(), displs.data(), MPI_UINT32_T, comm);
    auto work = [&](u32 k) { return u64(uses[k]) * lengths[k]; };
    std::vector<u64> cold(P, 0);
    for (int q = 0; q < P; q++)
        for (u32 k = block_start(K, q, P); k < block_start(K, q + 1, P); k++) cold[q] += work(k);
    h.total = std::accumulate(cold.begin(), cold.end(), u64(0));
    h.before = h.after = *std::max_element(cold.begin(), cold.end());
    const u64 target = (h.total + P - 1) / P;
    // Small or already balanced products do not justify constructing a shared cache.
    if (h.total < (u64(1) << 20) || h.before * 4 <= target * 5) return h;

    std::vector<u32> order;
    for (int q = 0; q < P; q++) {
        if (cold[q] <= target) continue;
        for (u32 k = block_start(K, q, P); k < block_start(K, q + 1, P); k++)
            if (work(k)) order.push_back(k);
    }
    // A heap avoids sorting every used B row when only a small hot subset is needed.
    auto less = [&](u32 x, u32 y) { return work(x) != work(y) ? work(x) < work(y) : x > y; };
    std::make_heap(order.begin(), order.end(), less);
    h.offset.assign(K, HotPlan::NONE);
    while (!order.empty() && *std::max_element(cold.begin(), cold.end()) > target) {
        std::pop_heap(order.begin(), order.end(), less);
        const u32 k = order.back();
        order.pop_back();
        const int q = row_owner(k, K, P);
        if (cold[q] <= target || h.entries + lengths[k] > cfg.bytes / sizeof(Entry)) continue;
        h.offset[k] = 0;
        h.entries += lengths[k];
        h.rows++;
        h.hot_products += work(k);
        cold[q] -= work(k);
    }
    if (!h.rows) {
        std::vector<u64>().swap(h.offset);
        return h;
    }

    std::vector<u64> row_work(M, 0);
    for (u32 il = 0; il < a.local_rows(); il++)
        for (u64 t = a.ptr[il]; t < a.ptr[il + 1]; t++)
            if (h.contains(a.data[t].col)) row_work[a.first + il] += lengths[a.data[t].col];
    MPI_Allreduce(MPI_IN_PLACE, row_work.data(), count_of(M), MPI_UINT64_T, MPI_SUM, comm);
    order.resize(M);
    std::iota(order.begin(), order.end(), u32(0));
    std::sort(order.begin(), order.end(), [&](u32 x, u32 y) {
        return row_work[x] != row_work[y] ? row_work[x] > row_work[y] : x < y;
    });
    h.worker.resize(M);
    for (u32 i : order) {
        int q = row_owner(i, M, P);
        for (int j = 0; j < P; j++) if (cold[j] < cold[q]) q = j;
        h.worker[i] = q;
        cold[q] += row_work[i];
    }
    h.after = *std::max_element(cold.begin(), cold.end());
    if (h.after * 10 >= h.before * 9) {
        // A coarse output row or too little cache can make this plan unprofitable.
        h.rows = h.entries = h.hot_products = 0;
        h.after = h.before;
        std::vector<u64>().swap(h.offset);
        std::vector<int>().swap(h.worker);
        return h;
    }
    h.starts.resize(P + 1);
    u64 pos = 0;
    for (int q = 0; q < P; q++) {
        h.starts[q] = pos;
        for (u32 k = block_start(K, q, P); k < block_start(K, q + 1, P); k++)
            if (h.contains(k)) {
                h.offset[k] = pos;
                pos += lengths[k];
            }
    }
    h.starts[P] = pos;
    h.length = std::move(lengths);
    return h;
}

class HotCache {
public:
    Entry *data = nullptr;
    int nodes = 0;

    void fill(const HotPlan &h, const Matrix &b, MPI_Comm comm, int rank, int P)
    {
        if (!h.entries) return;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &local_rank);
        int lead = rank;
        MPI_Bcast(&lead, 1, MPI_INT, 0, node);
        std::vector<int> all_leads(P), unique;
        MPI_Allgather(&lead, 1, MPI_INT, all_leads.data(), 1, MPI_INT, comm);
        for (int q = 0; q < P; q++) if (all_leads[q] == q) unique.push_back(q);
        nodes = int(unique.size());
        root.resize(P);
        for (int q = 0; q < P; q++)
            root[q] = int(std::lower_bound(unique.begin(), unique.end(), all_leads[q]) - unique.begin());
        MPI_Comm_split(comm, local_rank == 0 ? 0 : MPI_UNDEFINED, rank, &leaders);
        void *base = nullptr;
        MPI_Win_allocate_shared(local_rank == 0 ? MPI_Aint(h.entries * sizeof(Entry)) : 0,
                                sizeof(Entry), MPI_INFO_NULL, node, &base, &win);
        MPI_Aint size;
        int disp;
        MPI_Win_shared_query(win, 0, &size, &disp, &base);
        data = static_cast<Entry *>(base);
        MPI_Win_lock_all(0, win);
        for (u32 kl = 0; kl < b.local_rows(); kl++) {
            const u32 k = b.first + kl;
            if (h.contains(k))
                std::copy_n(b.data.data() + b.ptr[kl], h.length[k], data + h.offset[k]);
        }
        sync();
    }

    void share(const HotPlan &h)
    {
        if (!h.entries) return;
        // Packed ranges follow global B ownership, not node placement. Only leaders receive
        // network copies; other local ranks read the same shared cache after publication.
        if (local_rank == 0)
            for (size_t q = 0; q + 1 < h.starts.size(); q++) {
                const u64 n = h.starts[q + 1] - h.starts[q];
                if (n) MPI_Bcast(data + h.starts[q], count_of(n), MPI_UINT64_T, root[q], leaders);
            }
        sync();
    }

    void release()
    {
        if (win == MPI_WIN_NULL) return;
        MPI_Barrier(node);
        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
        if (leaders != MPI_COMM_NULL) MPI_Comm_free(&leaders);
        MPI_Comm_free(&node);
        data = nullptr;
    }

    ~HotCache() { if (!std::uncaught_exceptions()) release(); }

private:
    MPI_Comm node = MPI_COMM_NULL, leaders = MPI_COMM_NULL;
    MPI_Win win = MPI_WIN_NULL;
    int local_rank = 0;
    std::vector<int> root;

    void sync()
    {
        MPI_Win_sync(win);
        MPI_Barrier(node);
        MPI_Win_sync(win);
    }
};

}  // namespace

void multiply(const Matrix &a, const Matrix &b, Matrix &c)
{
    const MPI_Comm comm = MPI_COMM_WORLD;
    int P, r;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &r);
    Prof prof(comm);
    const u32 M = a.rows, K = a.cols, N = b.cols;
    const HotConfig cfg = hot_config(comm, r);
    const HotPlan hot = plan_hot(a, b, cfg, comm, P);
    prof.mark("hot-plan");
    HotCache cache;
    cache.fill(hot, b, comm, r, P);
    prof.mark("hot-local");
    cache.share(hot);
    prof.mark("hot-share");
    if (cfg.report && r == 0)
        std::fprintf(stderr, "[hw2-2-hot] enabled=%d selected_rows=%llu cache_bytes_per_node=%llu nodes=%d "
                             "total_products=%llu hot_products=%llu before_max=%llu planned_max=%llu\n",
                     cfg.enabled ? 1 : 0, (unsigned long long)hot.rows,
                     (unsigned long long)(hot.entries * sizeof(Entry)), cache.nodes,
                     (unsigned long long)hot.total, (unsigned long long)hot.hot_products,
                     (unsigned long long)hot.before, (unsigned long long)hot.after);
    auto destination = [&](u32 i, u32 k) { return hot.contains(k) ? hot.worker[i] : row_owner(k, K, P); };
    auto row_length = [&](u32 k) -> u64 {
        if (hot.contains(k)) return hot.length[k];
        const u32 kl = k - b.first;
        return b.ptr[kl + 1] - b.ptr[kl];
    };

    // Cold entries stay with B's owners; all hot entries of one output row share a worker.
    std::vector<u64> to(P, 0), from;
    for (u32 il = 0; il < a.local_rows(); il++)
        for (u64 e = a.ptr[il]; e < a.ptr[il + 1]; e++) to[destination(a.first + il, a.data[e].col)]++;
    std::vector<u64> at(P, 0);
    for (int q = 1; q < P; q++) at[q] = at[q - 1] + to[q - 1];
    std::vector<u64> send(a.data.size());
    for (u32 il = 0; il < a.local_rows(); il++) {
        const u64 i = a.first + il;
        for (u64 e = a.ptr[il]; e < a.ptr[il + 1]; e++) {
            const Entry x = a.data[e];
            send[at[destination(u32(i), x.col)]++] = i << 32 | u64(x.col) << 4 | x.val;
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
        for (u64 t = first[i]; t < first[i + 1]; t++) f += row_length(item[t] >> 4);
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
                const u32 k = item[t] >> 4;
                const u64 len = row_length(k);
                if (len) acc.add(item[t] & 15, hot.contains(k) ? cache.data + hot.offset[k]
                                                             : b.data.data() + b.ptr[k - b.first], len);
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
    cache.release();
    prof.mark("hot-release");
    prof.report();
}
