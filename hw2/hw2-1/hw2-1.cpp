// HW 2-1: multi-source shortest paths on a generated Kronecker graph, on the ranks of one node.
//
// Everything large lives in MPI shared-memory windows: they are exempt from the 1200 MiB private
// cap, and every rank on the node can read every other rank's part of the graph, so the work can
// be split by arcs rather than by who happens to own a vertex.
//
//   build     every rank generates M/P edges, drops self-loops, and sorts the two arcs of each
//             edge into a shared outbox by owner (mix(v) mod P, where mix is a bijection that
//             spreads scramble = 0's degree-laden low bits). Each owner then reads its arcs twice:
//             once to size its adjacency lists, once to fill them into the shared CSR.
//   traverse  all distinct sources at once, one distance value per round. Weights are integers
//             >= 1, so every (source, vertex) whose tentative distance equals the round's value d
//             is final: each is expanded exactly once. A round publishes the frontier to a shared
//             array, splits its arcs evenly over the ranks (a hub's list is cut too), and relaxes
//             with an atomic min; whoever lowers a distance files the vertex in its own 256-slot
//             ring of buckets (w <= 255, so pending values span at most 255).
//   results   each rank sums a contiguous slice of the vertices; MPI_Reduce brings it to rank 0.
//
// Phases are separated by MPI_Win_sync + MPI_Barrier, which is what makes one rank's plain stores
// into a window visible to the others. PP_TIMING=1 prints a per-phase breakdown to stderr, with
// time spent waiting at the barriers under "wait", so max vs min of a phase shows its imbalance;
// PP_BATCH=n caps the sources searched together (for testing the batching path).
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "graph_gen.hpp"

namespace {

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

constexpr u32 INF = 0xffffffffu;
constexpr u64 NONE = ~u64(0);
constexpr int VBITS = 26;                   // N <= 2^26, so a vertex id fits 26 bits
constexpr u64 VMASK = (u64(1) << VBITS) - 1;
constexpr int RING = 256;                   // > max weight, so pending values never alias

// Exact quotient and remainder of a 32-bit value by a runtime constant, without a divide
// (Lemire, Kaser and Kurz, "Faster remainder by direct computation"). Every arc goes through
// owner = v mod P and local = v / P at least twice.
struct FastDiv {
    u64 m;
    u32 d;
    explicit FastDiv(u32 d_) : m(d_ == 1 ? 0 : ~u64(0) / d_ + 1), d(d_) {}
    u32 div(u32 a) const { return d == 1 ? a : u32((__uint128_t(m) * a) >> 64); }
    u32 mod(u32 a) const { return d == 1 ? 0 : u32((__uint128_t(m * a) * d) >> 64); }
};

// A bijection on [0, 2^bits) that stirs every bit of a vertex id into its low bits. Raw Kronecker
// ids (scramble = 0) wear their degree in their bits: each 0 bit makes a vertex about three times
// heavier, so v mod P hands the owner of residue 0 most of the arcs (about 44% at P = 8). Owners
// and local slots come from mix(v) instead; the graph itself keeps the original ids. With
// scramble = 1 the generator's own bijection has already done this (v mod P gives 11.8-13.0% per
// owner at P = 8), so mix stays the identity and costs nothing.
class Mix {
public:
    Mix(int bits, bool on)
        : on(on), mask((u64(1) << bits) - 1), sh(std::max(1, bits / 2)), bits(bits),
          i1(inverse(K1)), i2(inverse(K2)), i3(inverse(K3))
    {
    }

    u32 fwd(u32 v) const
    {
        if (!on) return v;
        u64 x = v;
        x = (x * K1) & mask;
        x ^= x >> sh;
        x = (x * K2) & mask;
        x ^= x >> sh;
        return u32((x * K3) & mask);
    }

    u32 inv(u32 h) const
    {
        if (!on) return h;
        u64 x = h;
        x = unshift((x * i3) & mask);
        x = unshift((x * i2) & mask);
        return u32((x * i1) & mask);
    }

private:
    static constexpr u64 K1 = 0x9e3779b97f4a7c15ULL, K2 = 0xbf58476d1ce4e5b9ULL, K3 = 0x94d049bb133111ebULL;
    bool on;
    u64 mask;
    int sh, bits;
    u64 i1, i2, i3;

    // Inverse of y = x ^ (x >> sh): x = y ^ (y >> sh) ^ (y >> 2 sh) ^ ...
    u64 unshift(u64 y) const
    {
        u64 x = y;
        for (int s = sh; s < bits; s += sh) x ^= y >> s;
        return x;
    }

    // Odd a's inverse modulo 2^64 by Newton's iteration; it is also the inverse modulo 2^bits.
    static u64 inverse(u64 a)
    {
        u64 x = a;
        for (int i = 0; i < 6; i++) x *= 2 - a * x;
        return x;
    }
};

// The ranks that share memory, and the windows they share. alloc() is collective: rank 0 maps
// the whole window, everyone gets the same base pointer, and pages land on the NUMA node of
// whichever rank writes them first.
class Node {
public:
    MPI_Comm comm;
    int rank, size;

    explicit Node(MPI_Comm c) : comm(c)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    template <class T>
    T *alloc(u64 count)
    {
        const MPI_Aint bytes = rank == 0 ? MPI_Aint(std::max<u64>(count * sizeof(T), 64)) : 0;
        void *mine, *base;
        MPI_Win win;
        MPI_Win_allocate_shared(bytes, 1, MPI_INFO_NULL, comm, &mine, &win);
        MPI_Aint sz;
        int unit;
        MPI_Win_shared_query(win, 0, &sz, &unit, &base);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
        wins.push_back({base, win});
        return static_cast<T *>(base);
    }

    void release(const void *p)   // collective, like alloc
    {
        for (size_t i = 0; i < wins.size(); i++)
            if (wins[i].first == p) {
                MPI_Win_unlock_all(wins[i].second);
                MPI_Win_free(&wins[i].second);
                wins.erase(wins.begin() + i);
                return;
            }
    }

    void release_all()
    {
        while (!wins.empty()) release(wins.back().first);
    }

    // Orders this rank's window stores before what follows; paired with a barrier or a
    // collective it makes them visible to every other rank of the node.
    void fence()
    {
        for (auto &e : wins) MPI_Win_sync(e.second);
    }

    void sync()
    {
        fence();
        MPI_Barrier(comm);
        fence();
    }

private:
    std::vector<std::pair<void *, MPI_Win>> wins;
};

// Per-phase wall times, reduced over the ranks at the end (max and min) when PP_TIMING is set.
struct Prof {
    bool on = std::getenv("PP_TIMING") != nullptr;
    double last = MPI_Wtime();
    std::vector<std::pair<const char *, double>> phases;
    std::vector<std::pair<const char *, u64>> counts;

    void mark(const char *name)
    {
        if (!on) return;
        const double t = MPI_Wtime();
        for (auto &ph : phases)
            if (ph.first == name) { ph.second += t - last; last = t; return; }
        phases.push_back({name, t - last});
        last = t;
    }

    void count(const char *name, u64 v)
    {
        if (!on) return;
        for (auto &c : counts)
            if (c.first == name) { c.second += v; return; }
        counts.push_back({name, v});
    }

    void report(MPI_Comm comm, int rank)
    {
        if (!on) return;
        for (auto &ph : phases) {
            double mx = 0, mn = 0;
            MPI_Reduce(&ph.second, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&ph.second, &mn, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
            if (rank == 0) std::fprintf(stderr, "[hw2-1] %-10s max %9.4f s  min %9.4f s\n", ph.first, mx, mn);
        }
        for (auto &c : counts) {
            u64 sum = 0, mx = 0;
            MPI_Reduce(&c.second, &sum, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
            MPI_Reduce(&c.second, &mx, 1, MPI_UINT64_T, MPI_MAX, 0, comm);
            if (rank == 0)
                std::fprintf(stderr, "[hw2-1] %-10s sum %llu  max/rank %llu\n", c.first, (unsigned long long)sum,
                             (unsigned long long)mx);
        }
    }
};

// Adjacency of v: col/w[start, start + deg).
struct Adj {
    u32 start, deg;
};

struct Graph {
    Adj *adj;   // [N]
    u32 *col;   // [arcs]
    u8 *w;      // [arcs]
    u64 arcs;
};

// Generates this rank's slice of the edges and builds the shared CSR. Vertex v belongs to owner
// mix(v) mod P, in local slot mix(v) / P.
Graph build(Node &node, const GraphParams &p, const FastDiv &fd, Prof &prof)
{
    const int P = node.size, r = node.rank;
    const u64 N = num_vertices(p), M = num_edges(p);
    const Mix mix(p.scale, p.scramble == 0);

    // Generate edges [M r / P, M (r + 1) / P), self-loops dropped, packed (u | v << 26 | w << 52),
    // into this rank's stretch of a shared window: M / P edges privately would be 1 GiB at
    // scale 26 with P = 8, too close to the 1200 MiB cap.
    const u64 e0 = M * r / P, e1 = M * (r + 1) / P;
    u64 *edges = node.alloc<u64>(M);
    u64 kept = 0;                      // edges[e0, e0 + kept)
    std::vector<u64> to_owner(P, 0);   // arcs this rank sends to each owner
    {
        constexpr u64 CH = 1 << 14;
        std::vector<u32> uu(CH), vv(CH);
        std::vector<u8> ww(CH);
        for (u64 i = e0; i < e1; i += CH) {
            const u64 n = std::min(CH, e1 - i);
            generate_edges(p, i, i + n, uu.data(), vv.data(), ww.data());
            for (u64 j = 0; j < n; j++) {
                if (uu[j] == vv[j]) continue;
                edges[e0 + kept++] = u64(uu[j]) | u64(vv[j]) << VBITS | u64(ww[j]) << 52;
                to_owner[fd.mod(mix.fwd(uu[j]))]++;
                to_owner[fd.mod(mix.fwd(vv[j]))]++;
            }
        }
    }
    prof.mark("generate");

    // Outbox: rank q's segment, and inside it one run per owner. sent[q * P + o] = arcs q -> o.
    // An arc is packed (mix(src) | dst << 26 | w << 52): the owner needs src only as a slot.
    std::vector<u64> sent(u64(P) * P);
    MPI_Allgather(to_owner.data(), P, MPI_UINT64_T, sent.data(), P, MPI_UINT64_T, node.comm);
    std::vector<u64> run_begin(u64(P) * P);   // [q * P + o]: where q's run for o starts
    u64 total = 0;
    for (int q = 0; q < P; q++)
        for (int o = 0; o < P; o++) {
            run_begin[u64(q) * P + o] = total;
            total += sent[u64(q) * P + o];
        }
    u64 *outbox = node.alloc<u64>(total);
    prof.mark("wait");
    {
        std::vector<u64> cur(run_begin.begin() + u64(r) * P, run_begin.begin() + u64(r + 1) * P);
        for (u64 i = e0; i < e0 + kept; i++) {
            const u64 e = edges[i], w = (e >> 52) << 52;
            const u32 u = u32(e & VMASK), v = u32((e >> VBITS) & VMASK);
            const u32 hu = mix.fwd(u), hv = mix.fwd(v);
            outbox[cur[fd.mod(hu)]++] = u64(hu) | u64(v) << VBITS | w;   // u -> v
            outbox[cur[fd.mod(hv)]++] = u64(hv) | u64(u) << VBITS | w;   // v -> u
        }
    }
    prof.mark("scatter");
    node.sync();
    node.release(edges);
    prof.mark("wait");

    // Owner r: slots h = r + li * P, vertex mix^-1(h). First pass sizes their lists.
    const u64 nloc = N > u64(r) ? (N - 1 - r) / P + 1 : 0;
    std::vector<u32> pos(nloc, 0);
    auto for_my_arcs = [&](auto &&f) {
        for (int q = 0; q < P; q++) {
            const u64 *a = outbox + run_begin[u64(q) * P + r];
            const u64 n = sent[u64(q) * P + r];
            for (u64 i = 0; i < n; i++) f(a[i]);
        }
    };
    for_my_arcs([&](u64 a) { pos[fd.div(u32(a & VMASK))]++; });
    u64 mine = 0;
    for (u64 li = 0; li < nloc; li++) mine += pos[li];
    prof.mark("count");
    std::vector<u64> owned(P);
    MPI_Allgather(&mine, 1, MPI_UINT64_T, owned.data(), 1, MPI_UINT64_T, node.comm);
    prof.mark("wait");
    u64 base = 0, arcs = 0;
    for (int q = 0; q < P; q++) {
        if (q < r) base += owned[q];
        arcs += owned[q];
    }

    Graph g;
    g.arcs = arcs;
    g.adj = node.alloc<Adj>(N);
    g.col = node.alloc<u32>(arcs);
    g.w = node.alloc<u8>(arcs);
    for (u64 li = 0, at = base; li < nloc; li++) {
        g.adj[mix.inv(u32(r + li * P))] = {u32(at), pos[li]};
        const u64 d = pos[li];
        pos[li] = u32(at);   // now the fill cursor
        at += d;
    }
    prof.mark("alloc");

    // Second pass fills them.
    for_my_arcs([&](u64 a) {
        const u32 at = pos[fd.div(u32(a & VMASK))]++;
        g.col[at] = u32((a >> VBITS) & VMASK);
        g.w[at] = u8(a >> 52);
    });
    std::vector<u32>().swap(pos);
    prof.mark("fill");
    node.sync();
    node.release(outbox);
    prof.mark("wait");
    return g;
}

// Atomic min on a window word; true if this call lowered it.
inline bool lower(u32 *slot, u32 nd)
{
    u32 cur = __atomic_load_n(slot, __ATOMIC_RELAXED);
    while (nd < cur)
        if (__atomic_compare_exchange_n(slot, &cur, nd, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return true;
    return false;
}

// One batch of distinct sources, searched together. Items are packed (s << 26 | v).
class Search {
public:
    Search(Node &node, const Graph &g, u64 N, int cap, Prof &prof)
        : node(node), g(g), N(N), prof(prof), P(node.size), r(node.rank)
    {
        // A vertex is settled at most once per source, so N * cap bounds one round's frontier;
        // the coarse prefix keeps one entry per 64 items plus two per rank.
        dist = node.alloc<u32>(N * cap);
        front = node.alloc<u32>(N * cap);
        coarse = node.alloc<u64>(N * cap / 64 + 2 * u64(P) + 1);
        info.resize(3 * u64(P));
        off.resize(P);
        coff.resize(P);
        arc0.resize(P + 1);
    }

    // Distances of srcs[0..n) into dist[s * N + v]; INF where unreachable.
    void run(const u32 *srcs, int n)
    {
        sb = n;
        const u64 len = N * u64(n), lo = len * r / P, hi = len * (r + 1) / P;
        std::fill(dist + lo, dist + hi, INF);
        node.sync();
        for (int s = 0; s < n; s++)
            if (s % P == r) {
                dist[u64(s) * N + srcs[s]] = 0;
                ring[0].push_back(u32(s) << VBITS | srcs[s]);
            }

        u64 next = 0;   // every pending value lies in [next, next + 255]
        u64 rounds = 0, scanned = 0;
        for (;;) {
            // My smallest pending value whose bucket still holds a valid item. Stale items
            // (lowered again since they were filed) are dropped on the way.
            u64 mine[3] = {NONE, 0, 0};
            for (u64 x = next; x < next + RING; x++) {
                std::vector<u32> &b = ring[x % RING];
                if (b.empty()) continue;
                size_t k = 0;
                u64 a = 0;
                for (const u32 it : b) {
                    const u32 v = it & VMASK;
                    if (__atomic_load_n(&dist[u64(it >> VBITS) * N + v], __ATOMIC_RELAXED) == x) {
                        b[k++] = it;
                        a += g.adj[v].deg;
                    }
                }
                b.resize(k);
                if (k) {
                    mine[0] = x;
                    mine[1] = k;
                    mine[2] = a;
                    break;
                }
            }
            node.fence();
            MPI_Allgather(mine, 3, MPI_UINT64_T, info.data(), 3, MPI_UINT64_T, node.comm);
            node.fence();
            u64 d = NONE;
            for (int q = 0; q < P; q++) d = std::min(d, info[3 * q]);
            if (d == NONE) break;

            // Ranks holding value d publish their items; the rest sit this round out.
            u64 items = 0, chunks = 0;
            arc0[0] = 0;
            for (int q = 0; q < P; q++) {
                const bool in = info[3 * q] == d;
                cnt_of(q) = in ? info[3 * q + 1] : 0;
                arcs_of(q) = in ? info[3 * q + 2] : 0;
                off[q] = items;
                coff[q] = chunks;
                items += cnt_of(q);
                chunks += cnt_of(q) ? (cnt_of(q) + 63) / 64 + 1 : 0;
                arc0[q + 1] = arc0[q] + arcs_of(q);
            }
            if (mine[0] == d) {
                std::vector<u32> &b = ring[d % RING];
                u64 a = 0;
                for (size_t j = 0; j < b.size(); j++) {
                    if (j % 64 == 0) coarse[coff[r] + j / 64] = a;
                    front[off[r] + j] = b[j];
                    a += g.adj[b[j] & VMASK].deg;
                }
                coarse[coff[r] + (b.size() + 63) / 64] = a;
                b.clear();
            }
            node.sync();

            const u64 T = arc0[P];
            relax_range(u32(d), T * r / P, T * (r + 1) / P);
            next = d + 1;
            rounds++;
            scanned += T * (r + 1) / P - T * r / P;
        }
        prof.count("rounds", r == 0 ? rounds : 0);
        prof.count("arcs", scanned);
        prof.count("pushes", pushes);
        pushes = 0;
    }

    const u32 *distances() const { return dist; }

private:
    Node &node;
    const Graph &g;
    const u64 N;
    Prof &prof;
    const int P, r;
    int sb = 0;
    u32 *dist, *front;
    u64 *coarse;
    std::vector<u32> ring[RING];
    std::vector<u64> info, off, coff, arc0;
    u64 pushes = 0;

    // info is reused after the allgather: [3q + 1] items and [3q + 2] arcs of q this round.
    u64 &cnt_of(int q) { return info[3 * u64(q) + 1]; }
    u64 &arcs_of(int q) { return info[3 * u64(q) + 2]; }

    // Item index and arc offset of global arc t (arc0[q] <= t < arc0[q + 1]) in this round.
    void locate(u64 t, int &q, u64 &i, u64 &o) const
    {
        q = 0;
        while (arc0[q + 1] <= t) q++;
        const u64 rel = t - arc0[q];
        const u64 *c = coarse + coff[q];
        u64 lo = 0, hi = (info[3 * u64(q) + 1] + 63) / 64;   // c[hi] = all arcs of q > rel
        while (hi - lo > 1) {
            const u64 mid = (lo + hi) / 2;
            if (c[mid] <= rel) lo = mid; else hi = mid;
        }
        u64 acc = c[lo];
        i = lo * 64;
        for (;;) {
            const u32 dg = g.adj[front[off[q] + i] & VMASK].deg;
            if (acc + dg > rel) break;
            acc += dg;
            i++;
        }
        o = rel - acc;
    }

    void relax_range(u32 d, u64 lo, u64 hi)
    {
        if (lo >= hi) return;
        int q;
        u64 i, o;
        locate(lo, q, i, o);
        u64 left = hi - lo;
        while (left) {
            if (i == info[3 * u64(q) + 1]) {   // end of q's items: on to the next segment
                q++;
                i = 0;
                continue;
            }
            const u32 it = front[off[q] + i];
            const Adj a = g.adj[it & VMASK];
            const u64 take = std::min<u64>(a.deg - o, left);
            relax(it >> VBITS, d, u64(a.start) + o, take);
            left -= take;
            o = 0;
            i++;
        }
    }

    void relax(u32 s, u32 d, u64 e0, u64 n)
    {
        u32 *ds = dist + u64(s) * N;
        const u32 *col = g.col + e0;
        const u8 *w = g.w + e0;
        constexpr u64 PF = 16;
        for (u64 e = 0; e < n; e++) {
            if (e + PF < n) __builtin_prefetch(ds + col[e + PF]);
            const u32 y = col[e], nd = d + w[e];
            if (lower(ds + y, nd)) {
                ring[nd % RING].push_back(s << VBITS | y);
                pushes++;
            }
        }
    }
};

}  // namespace

void run(const GraphParams &p, const std::vector<uint32_t> &sources, std::vector<SourceResult> &results)
{
    // Shared windows need one node: the ranks on rank 0's node do the work, any others idle.
    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, wrank, MPI_INFO_NULL, &shared);
    int lowest;
    MPI_Allreduce(&wrank, &lowest, 1, MPI_INT, MPI_MIN, shared);
    MPI_Comm_free(&shared);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, lowest == 0 ? 0 : MPI_UNDEFINED, wrank, &comm);
    if (comm == MPI_COMM_NULL) return;

    {
        Node node(comm);
        Prof prof;
        const int P = node.size, r = node.rank;
        const FastDiv fd{u32(P)};
        const u64 N = num_vertices(p);

        const Graph g = build(node, p, fd, prof);

        // Distinct sources, first-occurrence order; repeats copy their twin's result.
        std::vector<u32> uniq;
        std::vector<int> which(sources.size());
        for (size_t i = 0; i < sources.size(); i++) {
            const auto it = std::find(uniq.begin(), uniq.end(), sources[i]);
            which[i] = int(it - uniq.begin());
            if (it == uniq.end()) uniq.push_back(sources[i]);
        }

        // dist + frontier + coarse prefix cost about 8.125 N bytes per source searched together.
        const u64 budget = u64(20) << 30;
        int cap = int(std::max<u64>(1, std::min<u64>(uniq.size(), budget / (N * 8 + N / 8 + 1))));
        if (const char *e = std::getenv("PP_BATCH")) cap = std::max(1, std::min(cap, std::atoi(e)));
        prof.count("batches", r == 0 ? (uniq.size() + cap - 1) / cap : 0);

        Search search(node, g, N, cap, prof);
        const std::vector<u32> samples = r == 0 ? sample_vertices(p) : std::vector<u32>();
        std::vector<SourceResult> found(uniq.size());
        prof.mark("setup");

        for (size_t b0 = 0; b0 < uniq.size(); b0 += cap) {
            const int n = int(std::min<size_t>(cap, uniq.size() - b0));
            search.run(uniq.data() + b0, n);
            prof.mark("traverse");

            // n_reached, checksum, digest1, digest2 over my slice of the vertices.
            const u32 *dist = search.distances();
            std::vector<u64> sums(4 * u64(n), 0), total(4 * u64(n), 0);
            const u64 v0 = N * r / P, v1 = N * (r + 1) / P;
            for (int s = 0; s < n; s++) {
                const u32 *ds = dist + u64(s) * N;
                u64 cnt = 0, cs = 0, d1 = 0, d2 = 0;
                for (u64 v = v0; v < v1; v++) {
                    const u32 dv = ds[v];
                    if (dv == INF) continue;
                    cnt++;
                    cs += u64(dv) * (v + 1);
                    d1 += distance_digest(u32(v), dv, 0);
                    d2 += distance_digest(u32(v), dv, 1);
                }
                sums[4 * s] = cnt;
                sums[4 * s + 1] = cs;
                sums[4 * s + 2] = d1;
                sums[4 * s + 3] = d2;
            }
            MPI_Reduce(sums.data(), total.data(), 4 * n, MPI_UINT64_T, MPI_SUM, 0, comm);
            if (r == 0)
                for (int s = 0; s < n; s++) {
                    SourceResult &res = found[b0 + s];
                    res.n_reached = int64_t(total[4 * s]);
                    res.checksum = total[4 * s + 1];
                    res.digest1 = total[4 * s + 2];
                    res.digest2 = total[4 * s + 3];
                    const u32 *ds = dist + u64(s) * N;
                    res.sample_dist.resize(samples.size());
                    for (size_t k = 0; k < samples.size(); k++)
                        res.sample_dist[k] = ds[samples[k]] == INF ? -1 : int32_t(ds[samples[k]]);
                }
            node.sync();   // rank 0 is done reading before the next batch overwrites dist
            prof.mark("results");
        }

        if (r == 0)
            for (size_t i = 0; i < sources.size(); i++) {
                const SourceResult &f = found[which[i]];
                results[i].n_reached = f.n_reached;
                results[i].checksum = f.checksum;
                results[i].digest1 = f.digest1;
                results[i].digest2 = f.digest2;
                std::copy(f.sample_dist.begin(), f.sample_dist.end(), results[i].sample_dist.begin());
            }

        if (prof.on && r == 0)
            std::fprintf(stderr, "[hw2-1] P=%d N=%llu M=%llu arcs=%llu sources=%zu distinct=%zu batch=%d\n", P,
                         (unsigned long long)N, (unsigned long long)num_edges(p), (unsigned long long)g.arcs,
                         sources.size(), uniq.size(), cap);
        prof.report(comm, r);
        node.release_all();
    }
    MPI_Comm_free(&comm);
}
