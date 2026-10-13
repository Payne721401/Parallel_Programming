// HW 2-1: multi-source shortest paths on a generated Kronecker graph, on the ranks of one node.
//
// Everything large lives in MPI shared-memory windows: they are exempt from the 1200 MiB private
// cap, and every rank on the node can read every other rank's part of the graph, so the work can
// be split by arcs rather than by who happens to own a vertex.
//
//   build     every rank generates M/P edges, drops self-loops, and sorts the two arcs of each
//             edge into a shared outbox by owner (mix(v) mod P, where mix is a bijection that
//             spreads scramble = 0's degree-laden low bits). Each owner then reads its arcs twice:
//             once to size its adjacency lists, once to fill them into the shared CSR. The edges
//             come from FastGen, a bit-exact rendition of graph_gen.cpp's generator that is
//             checked against the official one before it is trusted.
//   slots     between the passes, vertices get new numbers ("slots"), heaviest first, and the
//             isolated ones none: the arc ends that relaxations read cluster in a few MB of each
//             distance array instead of spreading over all of it, and the arrays shrink by the
//             isolated share (about half the vertices at scale 23-24).
//   memory    a window is one shared-memory object, and first touches of its pages queue on that
//             object (measured, 8 ranks at once: 7.1 us a page in one window, 2.5 in a window
//             each). So what a rank alone writes sits in windows of its own, and since a page
//             already touched costs no fault at all, they are reused by lifetime: rank r's arena
//             holds its generated edges, then its part of the CSR, whose col (4 B an arc, two arcs
//             an edge) lands on the pages the edges (8 B) took. Its outbox is a window of its own;
//             only the small per-vertex tables and the search's arrays share one. Nothing is
//             freed before the end: a window's pages go back serially, 0.1-0.2 s per GB.
//   traverse  all distinct sources at once, one distance value per round. Weights are integers
//             >= 1, so every (source, vertex) whose tentative distance equals the round's value d
//             is final: each is expanded exactly once. A round publishes the frontier to a shared
//             array, splits its arcs evenly over the ranks (a hub's list is cut too), and relaxes
//             with an atomic min; whoever lowers a distance files the vertex in its own 256-slot
//             ring of buckets (w <= 255, so pending values span at most 255). About 60% of
//             relaxations (measured, scale 24-26) reach a target already final, which nothing can
//             lower: a bitmap of the final (source, slot) pairs, 1/32 the size of the distances
//             and set as the frontier is published, lets those skip reading the distance at all.
//   results   each rank sums a contiguous slice of the vertices; MPI_Reduce brings it to rank 0.
//
// Phases are separated by MPI_Win_sync + MPI_Barrier, which is what makes one rank's plain stores
// into a window visible to the others.
//
// Measurement exists only in a build with -DHW21_PROF (tests/run_hw2-1.sh builds it as
// hw2-1-prof); the judge's build compiles none of it. That build prints, per phase, the slowest
// and fastest rank's time and the most page faults any rank took, time spent waiting at barriers
// under "wait", timings nested inside phases (window alloc/free, the traversal's parts) and
// counters, among them how relaxations end: on a target already final (relax-final, of which
// relax-bitmap the bitmap caught), on one whose tentative distance is no larger (relax-nogain),
// or lowering it (pushes). There, PP_STATS=1 adds the graph's shape, PP_PRETOUCH=1 touches the
// CSR's pages before filling it, and PP_PROBE=1 times the fill's reads alone (cursors, far ends'
// slots) before the fill. In any build, PP_BATCH=n caps the sources searched together,
// PP_FASTGEN=0 uses the official generator, PP_RELABEL=0 keeps the original vertex numbers and
// PP_BITMAP=0 reads every target's distance.
#include <mpi.h>
#ifdef HW21_PROF
#include <sys/resource.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__)
#include <immintrin.h>
#define HW21_AVX512 1
#endif

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

#ifdef HW21_PROF
// Per-phase wall time and minor page faults (each first touch of a fresh page, shared windows
// included), timings nested inside phases, and counters; reduced over the ranks at the end.
class Prof {
public:
    static constexpr bool on = true;

    // Closes the phase that has run since the previous mark.
    void mark(const char *name)
    {
        const double t = MPI_Wtime();
        const u64 f = faults();
        Item &it = find(phases, name);
        it.t += t - last;
        it.faults += f - flt;
        last = t;
        flt = f;
    }

    // A timing inside the current phase, reported apart from the phases.
    void sub(const char *what, const char *tag, double t) { find(subs, std::string(what) + tag).t += t; }

    void count(const char *name, u64 v)
    {
        for (auto &c : counts)
            if (c.first == name) { c.second += v; return; }
        counts.push_back({name, v});
    }

    void report(MPI_Comm comm, int rank)
    {
        for (int kind = 0; kind < 2; kind++)
            for (Item &it : kind ? subs : phases) {
                double mx = 0, mn = 0;
                u64 fmax = 0;
                MPI_Reduce(&it.t, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
                MPI_Reduce(&it.t, &mn, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
                MPI_Reduce(&it.faults, &fmax, 1, MPI_UINT64_T, MPI_MAX, 0, comm);
                if (rank != 0) continue;
                std::fprintf(stderr, "[hw2-1] %-14s max %9.4f s  min %9.4f s", it.name.c_str(), mx, mn);
                if (!kind) std::fprintf(stderr, "  faults %llu", (unsigned long long)fmax);
                std::fprintf(stderr, "\n");
            }
        for (auto &c : counts) {
            u64 sum = 0, mx = 0;
            MPI_Reduce(&c.second, &sum, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
            MPI_Reduce(&c.second, &mx, 1, MPI_UINT64_T, MPI_MAX, 0, comm);
            if (rank == 0)
                std::fprintf(stderr, "[hw2-1] %-14s sum %llu  max/rank %llu\n", c.first.c_str(),
                             (unsigned long long)sum, (unsigned long long)mx);
        }
    }

private:
    struct Item {
        std::string name;
        double t = 0;
        u64 faults = 0;
    };
    std::vector<Item> phases, subs;
    std::vector<std::pair<std::string, u64>> counts;
    double last = MPI_Wtime();
    u64 flt = faults();

    static u64 faults()
    {
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        return u64(ru.ru_minflt);
    }

    static Item &find(std::vector<Item> &v, const std::string &name)
    {
        for (Item &it : v)
            if (it.name == name) return it;
        v.push_back(Item{name});
        return v.back();
    }
};
#else
// The judge's build: measurement compiled out, every call an empty inline.
class Prof {
public:
    static constexpr bool on = false;
    void mark(const char *) {}
    void sub(const char *, const char *, double) {}
    void count(const char *, u64) {}
    void report(MPI_Comm, int) {}
};
#endif

// MPI_Wtime in the profiling build, nothing in the judge's.
inline double tick() { return Prof::on ? MPI_Wtime() : 0.0; }

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

// graph_gen.cpp's mix(), the splitmix64 finalizer.
inline u64 mix64(u64 x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// graph_gen.cpp's generate_edge, bit for bit, several times faster. The official one rederives the
// seed's hashes and the scramble keys for every edge (writes through uint8_t* may alias p, so the
// compiler cannot hoist them) and walks the Kronecker descent one level at a time. This one
// hoists them, decides eight levels per 64-bit word with byte-wise compares, and with AVX-512
// runs eight edges at once. fastgen_matches() checks it against the official generator first.
class FastGen {
public:
    explicit FastGen(const GraphParams &p)
        : scale(p.scale), sh(p.scale / 2 > 0 ? p.scale / 2 : 1), scr(p.scramble != 0), ms(mix64(p.seed)),
          mw(mix64(p.seed ^ 0x77e19470ULL)), mask((u64(1) << p.scale) - 1)
    {
        const u64 s = p.seed ^ 0x5c2a3b1eULL;   // scramble_id's seed
        k1 = mix64(s ^ 0x1111ULL) | 1;
        k2 = mix64(s ^ 0x2222ULL) | 1;
        k3 = mix64(s ^ 0x3333ULL) | 1;
    }

    void edge(u64 i, u32 &u, u32 &v, u8 &w) const
    {
        u64 state = mix64(ms ^ (i * HK));   // hash2(seed, i)
        u32 a = 0, b = 0;
        for (int base = 0; base < scale; base += 8) {
            if (base) state = mix64(state);
            // Per byte r of the word: bit_a = r >= 195, bit_b = (r >= 146 && r < 195) || r >= 243.
            const u64 g146 = ge(state, 0x1212121212121212ULL), g195 = ge(state, 0x4343434343434343ULL),
                      g243 = ge(state, 0x7373737373737373ULL);
            u32 am = pack8(g195), bm = pack8((g146 & ~g195) | g243);
            if (scale - base < 8) {
                am &= (1u << (scale - base)) - 1;
                bm &= (1u << (scale - base)) - 1;
            }
            a |= am << base;
            b |= bm << base;
        }
        if (scr) {
            a = scramble(a);
            b = scramble(b);
        }
        u = a;
        v = b;
        const u64 lo = std::min(a, b), hi = std::max(a, b);
        w = u8(1 + mix64(mw ^ (((hi << 32) | lo) * HK)) % 255);
    }

    // Edges [i0, i0 + n) into three arrays, as generate_edges does.
    void edges(u64 i0, u64 n, u32 *u, u32 *v, u8 *w) const
    {
        u64 j = 0;
#ifdef HW21_AVX512
        for (; j + 8 <= n; j += 8) edges8(i0 + j, u + j, v + j, w + j);
#endif
        for (; j < n; j++) edge(i0 + j, u[j], v[j], w[j]);
    }

private:
    static constexpr u64 HK = 0xd6e8feb86659fd93ULL;   // hash2's multiplier
    int scale, sh;
    bool scr;
    u64 ms, mw, mask, k1, k2, k3;

    u32 scramble(u64 x) const
    {
        x = (x * k1) & mask;
        x ^= x >> sh;
        x = (x * k2) & mask;
        x ^= x >> sh;
        return u32((x * k3) & mask);
    }

    // Top bit of each byte of x set where that byte >= c, for c >= 128; l = c - 128 per byte.
    static u64 ge(u64 x, u64 l)
    {
        constexpr u64 H = 0x8080808080808080ULL;
        return x & ((x | H) - l) & H;
    }

    // The eight byte-top bits of m as one byte, byte j's to bit j.
    static u32 pack8(u64 m) { return u32(((m >> 7) * 0x0102040810204080ULL) >> 56); }

#ifdef HW21_AVX512
    static __m512i mixv(__m512i x)
    {
        x = _mm512_add_epi64(x, _mm512_set1_epi64(0x9e3779b97f4a7c15ULL));
        x = _mm512_mullo_epi64(_mm512_xor_si512(x, _mm512_srli_epi64(x, 30)), _mm512_set1_epi64(0xbf58476d1ce4e5b9ULL));
        x = _mm512_mullo_epi64(_mm512_xor_si512(x, _mm512_srli_epi64(x, 27)), _mm512_set1_epi64(0x94d049bb133111ebULL));
        return _mm512_xor_si512(x, _mm512_srli_epi64(x, 31));
    }

    void edges8(u64 i0, u32 *u, u32 *v, u8 *w) const
    {
        const __m512i idx = _mm512_add_epi64(_mm512_set1_epi64(i0), _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
        __m512i state = mixv(_mm512_xor_si512(_mm512_set1_epi64(ms), _mm512_mullo_epi64(idx, _mm512_set1_epi64(HK))));
        __m512i a = _mm512_setzero_si512(), b = _mm512_setzero_si512();
        const __m512i ff = _mm512_set1_epi64(0xff), c146 = _mm512_set1_epi64(146), c195 = _mm512_set1_epi64(195),
                      c243 = _mm512_set1_epi64(243);
        for (int level = 0; level < scale; level++) {
            if (level && (level & 7) == 0) state = mixv(state);
            const __m512i r = _mm512_and_si512(_mm512_srli_epi64(state, 8 * (level & 7)), ff);
            const __mmask8 ma = _mm512_cmpge_epu64_mask(r, c195);
            const __mmask8 mb = (_mm512_cmpge_epu64_mask(r, c146) & ~ma) | _mm512_cmpge_epu64_mask(r, c243);
            const __m512i bit = _mm512_set1_epi64(u64(1) << level);
            a = _mm512_mask_or_epi64(a, ma, a, bit);
            b = _mm512_mask_or_epi64(b, mb, b, bit);
        }
        if (scr) {
            const __m512i m = _mm512_set1_epi64(mask), q1 = _mm512_set1_epi64(k1), q2 = _mm512_set1_epi64(k2),
                          q3 = _mm512_set1_epi64(k3);
            for (__m512i *x : {&a, &b}) {
                __m512i t = _mm512_and_si512(_mm512_mullo_epi64(*x, q1), m);
                t = _mm512_xor_si512(t, _mm512_srli_epi64(t, sh));
                t = _mm512_and_si512(_mm512_mullo_epi64(t, q2), m);
                t = _mm512_xor_si512(t, _mm512_srli_epi64(t, sh));
                *x = _mm512_and_si512(_mm512_mullo_epi64(t, q3), m);
            }
        }
        const __m512i lo = _mm512_min_epu64(a, b), hi = _mm512_max_epu64(a, b);
        const __m512i key = _mm512_or_si512(_mm512_slli_epi64(hi, 32), lo);
        const __m512i h = mixv(_mm512_xor_si512(_mm512_set1_epi64(mw), _mm512_mullo_epi64(key, _mm512_set1_epi64(HK))));
        // h mod 255 = (sum of h's bytes) mod 255, since 256 = 1 (mod 255).
        __m512i s = _mm512_sad_epu8(h, _mm512_setzero_si512());                  // <= 2040
        s = _mm512_add_epi64(_mm512_and_si512(s, ff), _mm512_srli_epi64(s, 8));  // <= 262
        s = _mm512_mask_sub_epi64(s, _mm512_cmpge_epu64_mask(s, _mm512_set1_epi64(255)), s, _mm512_set1_epi64(255));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(u), _mm512_cvtepi64_epi32(a));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(v), _mm512_cvtepi64_epi32(b));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(w), _mm512_cvtepi64_epi8(_mm512_add_epi64(s, _mm512_set1_epi64(1))));
    }
#endif
};

// FastGen against the official generator: this rank's first edges (the vector path) and 64 edges
// spread over the whole range (the scalar path). Any difference and the official one does the work.
bool fastgen_matches(const GraphParams &p, const FastGen &fg, u64 e0, u64 e1)
{
    const u64 M = num_edges(p), n = std::min<u64>(4096, e1 - e0);
    std::vector<u32> u0(n), v0(n), u1(n), v1(n);
    std::vector<u8> w0(n), w1(n);
    generate_edges(p, e0, e0 + n, u0.data(), v0.data(), w0.data());
    fg.edges(e0, n, u1.data(), v1.data(), w1.data());
    if (u0 != u1 || v0 != v1 || w0 != w1) return false;
    for (u64 k = 0; k < 64; k++) {
        const u64 i = (M - 1) * k / 63;
        u32 a, b, c, d;
        u8 x, y;
        generate_edge(p, i, a, b, x);
        fg.edge(i, c, d, y);
        if (a != c || b != d || x != y) return false;
    }
    return true;
}

// The ranks that share memory, and the windows they share. alloc() is collective: rank 0 maps
// the whole window, everyone gets the same base pointer, and pages land on the NUMA node of
// whichever rank writes them first.
class Node {
public:
    MPI_Comm comm;
    int rank, size;

    Node(MPI_Comm c, Prof &prof) : comm(c), prof(prof)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    template <class T>
    T *alloc(u64 count, const char *tag)
    {
        const double t0 = tick();
        const MPI_Aint bytes = rank == 0 ? MPI_Aint(std::max<u64>(count * sizeof(T), 64)) : 0;
        void *mine, *base;
        MPI_Win win;
        MPI_Win_allocate_shared(bytes, 1, MPI_INFO_NULL, comm, &mine, &win);
        MPI_Aint sz;
        int unit;
        MPI_Win_shared_query(win, 0, &sz, &unit, &base);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
        wins.push_back({base, win, tag});
        prof.sub("alloc:", tag, tick() - t0);
        return static_cast<T *>(base);
    }

    // One window per rank, rank j's holding j's bytes. Each window is its own shared-memory
    // object, so ranks first-touching their own parts never queue on one object's page index
    // (8 ranks at once: 2.5 us a page, against 7.1 for one window). Returns every rank's part.
    std::vector<char *> alloc_each(u64 bytes, const char *tag)
    {
        const double t0 = tick();
        std::vector<char *> parts(size);
        for (int j = 0; j < size; j++) {
            void *mine, *base;
            MPI_Win win;
            MPI_Win_allocate_shared(rank == j ? MPI_Aint(std::max<u64>(bytes, 64)) : 0, 1, MPI_INFO_NULL, comm, &mine,
                                    &win);
            MPI_Aint sz;
            int unit;
            MPI_Win_shared_query(win, j, &sz, &unit, &base);
            MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
            wins.push_back({base, win, tag});
            parts[j] = static_cast<char *>(base);
        }
        prof.sub("alloc:", tag, tick() - t0);
        return parts;
    }

    void release_all()   // collective, like alloc
    {
        while (!wins.empty()) {
            const double t0 = tick();
            MPI_Win_unlock_all(wins.back().win);
            MPI_Win_free(&wins.back().win);
            prof.sub("free:", wins.back().tag, tick() - t0);
            wins.pop_back();
        }
    }

    // Orders this rank's window stores before what follows; paired with a barrier or a
    // collective it makes them visible to every other rank of the node.
    void fence()
    {
        for (auto &e : wins) MPI_Win_sync(e.win);
    }

    void sync()
    {
        fence();
        MPI_Barrier(comm);
        fence();
    }

private:
    struct Window {
        void *base;
        MPI_Win win;
        const char *tag;
    };
    Prof &prof;
    std::vector<Window> wins;
};

// Adjacency of a slot: its owner's col/w[start, start + deg).
struct Adj {
    u32 start, deg;
};

// The graph over slots. With relabeling (the default) the slots are degree-ordered and the
// isolated vertices have none; with PP_RELABEL=0 a vertex's slot is its own number. Each owner's
// adjacency lists sit in that owner's own window: col[own[s]] + adj[s].start.
struct Graph {
    u64 n;                  // slots: the non-isolated vertices, or N
    Adj *adj;               // [n]
    uint16_t *own;          // [n]: the slot's owner
    std::vector<u32 *> col; // per owner: the far ends' slots
    std::vector<u8 *> w;    // per owner: the weights
    u64 arcs;
    bool relabeled;
    u32 *newid;             // [N]: a vertex's slot, INF for an isolated one (relabeled only)
    u32 *oldid;             // [n]: a slot's vertex (relabeled only)

    u32 slot(u32 v) const { return relabeled ? newid[v] : v; }
    u32 vertex(u32 s) const { return relabeled ? oldid[s] : s; }
    const u32 *cols(u32 s) const { return col[own[s]] + adj[s].start; }
    const u8 *weights(u32 s) const { return w[own[s]] + adj[s].start; }
};

inline u64 page_up(u64 x) { return (x + 4095) & ~u64(4095); }

#ifdef HW21_PROF
// PP_STATS=1: the graph's shape at the real size. Owners look at their own vertices: isolated
// ones, duplicate arcs, leaves (one distinct neighbour), and per log2(degree) bucket the vertices
// and their arcs, i.e. how many of the arcs' far ends land on the heaviest vertices, which is what
// the degree-ordered slots keep together in the caches.
void graph_stats(Node &node, const Graph &g, u64 N, const Mix &mix, u64 nloc)
{
    const int P = node.size, r = node.rank;
    constexpr int B = 32;
    std::vector<u64> mine(4 + 2 * B, 0), all(4 + 2 * B, 0);   // isolated, dup, leaves, maxdeg | count[B] | arcs[B]
    std::vector<u32> tmp;
    for (u64 li = 0; li < nloc; li++) {
        const u32 s = g.slot(mix.inv(u32(r + li * P)));
        if (s == INF || !g.adj[s].deg) {
            mine[0]++;
            continue;
        }
        const Adj a = g.adj[s];
        tmp.assign(g.cols(s), g.cols(s) + a.deg);
        std::sort(tmp.begin(), tmp.end());
        const u64 uniq = u64(std::unique(tmp.begin(), tmp.end()) - tmp.begin());
        mine[1] += a.deg - uniq;
        mine[2] += uniq == 1;
        mine[3] = std::max<u64>(mine[3], a.deg);
        const int b = 31 - __builtin_clz(a.deg);
        mine[4 + b]++;
        mine[4 + B + b] += a.deg;
    }
    MPI_Reduce(mine.data(), all.data(), 3, MPI_UINT64_T, MPI_SUM, 0, node.comm);
    MPI_Reduce(&mine[3], &all[3], 1, MPI_UINT64_T, MPI_MAX, 0, node.comm);
    MPI_Reduce(&mine[4], &all[4], 2 * B, MPI_UINT64_T, MPI_SUM, 0, node.comm);
    if (r != 0 || !g.arcs) return;
    const u64 live = N - all[0];
    std::fprintf(stderr,
                 "[hw2-1] stats: isolated %.1f%% of N, duplicate arcs %.1f%%, leaves %.1f%% of non-isolated, "
                 "max degree %llu (duplicates counted)\n",
                 100.0 * all[0] / N, 100.0 * all[1] / g.arcs, live ? 100.0 * all[2] / live : 0.0,
                 (unsigned long long)all[3]);
    std::fprintf(stderr, "[hw2-1] stats: arc ends on the heaviest vertices:");
    u64 cv = 0, ca = 0;
    for (int b = B - 1; b >= 0; b--) {
        if (!all[4 + b]) continue;
        cv += all[4 + b];
        ca += all[4 + B + b];
        std::fprintf(stderr, "  deg>=%u: %llu v %.0f%%", 1u << b, (unsigned long long)cv, 100.0 * ca / g.arcs);
        if (ca * 100 >= g.arcs * 99) break;
    }
    std::fprintf(stderr, "\n");
}
#endif

// Generates this rank's slice of the edges and builds the shared CSR over slots. Vertex v belongs
// to owner mix(v) mod P, in local position mix(v) / P.
Graph build(Node &node, const GraphParams &p, const FastDiv &fd, Prof &prof)
{
    const int P = node.size, r = node.rank;
    const u64 N = num_vertices(p), M = num_edges(p);
    const Mix mix(p.scale, p.scramble == 0);
    const char *re = std::getenv("PP_RELABEL");
    const bool relabel = !(re && re[0] == '0');

    // Rank q's arena: its edges (8 B each), then its part of the CSR (5 B an arc, about two arcs
    // an edge), col first so that it lands on the pages the edges took. 12 B an edge leaves room
    // for an owner a fifth heavier than average; untouched pages cost nothing.
    auto arena_bytes = [&](int q) { return page_up(12 * (M * (q + 1) / P - M * q / P)) + 4096; };
    const std::vector<char *> arena = node.alloc_each(arena_bytes(r), "arena");

    // Generate edges [M r / P, M (r + 1) / P), self-loops dropped, packed (u | v << 26 | w << 52).
    const u64 e0 = M * r / P, e1 = M * (r + 1) / P;
    u64 *const edges = reinterpret_cast<u64 *>(arena[r]);
    const FastGen fg(p);
    const char *fe = std::getenv("PP_FASTGEN");
    const bool fast = !(fe && fe[0] == '0') && fastgen_matches(p, fg, e0, e1);
    prof.count("fastgen-off", fast ? 0 : 1);
    u64 kept = 0;                      // edges[0, kept)
    std::vector<u64> to_owner(P, 0);   // arcs this rank sends to each owner
    {
        constexpr u64 CH = 1 << 14;
        std::vector<u32> uu(CH), vv(CH);
        std::vector<u8> ww(CH);
        for (u64 i = e0; i < e1; i += CH) {
            const u64 n = std::min(CH, e1 - i);
            if (fast)
                fg.edges(i, n, uu.data(), vv.data(), ww.data());
            else
                generate_edges(p, i, i + n, uu.data(), vv.data(), ww.data());
            for (u64 j = 0; j < n; j++) {
                if (uu[j] == vv[j]) continue;
                edges[kept++] = u64(uu[j]) | u64(vv[j]) << VBITS | u64(ww[j]) << 52;
                to_owner[fd.mod(mix.fwd(uu[j]))]++;
                to_owner[fd.mod(mix.fwd(vv[j]))]++;
            }
        }
    }
    prof.mark("generate");

    // Outbox: one window per producer q, holding q's runs for owners 0..P-1 in order.
    // sent[q * P + o] = arcs q -> o. An arc is packed (mix(src) | dst << 26 | w << 52): the
    // owner needs src only as a position.
    std::vector<u64> sent(u64(P) * P);
    MPI_Allgather(to_owner.data(), P, MPI_UINT64_T, sent.data(), P, MPI_UINT64_T, node.comm);
    std::vector<u64> run_begin(u64(P) * P);   // [q * P + o]: where q's run for o starts in q's window
    for (int q = 0; q < P; q++) {
        u64 at = 0;
        for (int o = 0; o < P; o++) {
            run_begin[u64(q) * P + o] = at;
            at += sent[u64(q) * P + o];
        }
    }
    const std::vector<char *> outbox = node.alloc_each(8 * kept * 2, "outbox");
    prof.mark("wait");
    {
        u64 *const mine = reinterpret_cast<u64 *>(outbox[r]);
        std::vector<u64> cur(run_begin.begin() + u64(r) * P, run_begin.begin() + u64(r + 1) * P);
        for (u64 i = 0; i < kept; i++) {
            const u64 e = edges[i], w = (e >> 52) << 52;
            const u32 u = u32(e & VMASK), v = u32((e >> VBITS) & VMASK);
            const u32 hu = mix.fwd(u), hv = mix.fwd(v);
            mine[cur[fd.mod(hu)]++] = u64(hu) | u64(v) << VBITS | w;   // u -> v
            mine[cur[fd.mod(hv)]++] = u64(hv) | u64(u) << VBITS | w;   // v -> u
        }
    }
    prof.mark("scatter");
    node.sync();   // my edges are dead from here: my arena waits for my part of the CSR
    prof.mark("wait");

    // Owner r: positions h = r + li * P, vertex mix^-1(h). First pass sizes their lists.
    const u64 nloc = N > u64(r) ? (N - 1 - r) / P + 1 : 0;
    std::vector<u32> pos(nloc, 0);
    auto for_my_arcs = [&](auto &&f) {
        for (int q = 0; q < P; q++) {
            const u64 *a = reinterpret_cast<const u64 *>(outbox[q]) + run_begin[u64(q) * P + r];
            const u64 n = sent[u64(q) * P + r];
            for (u64 i = 0; i < n; i++) f(a[i]);
        }
    };
    for_my_arcs([&](u64 a) { pos[fd.div(u32(a & VMASK))]++; });
    u64 mine = 0;
    u64 cls[32] = {};   // my non-isolated vertices per log2(degree)
    for (u64 li = 0; li < nloc; li++)
        if (pos[li]) {
            mine += pos[li];
            cls[31 - __builtin_clz(pos[li])]++;
        }
    prof.mark("count");
    std::vector<u64> owned(P), classes(32 * u64(P));
    MPI_Allgather(&mine, 1, MPI_UINT64_T, owned.data(), 1, MPI_UINT64_T, node.comm);
    MPI_Allgather(cls, 32, MPI_UINT64_T, classes.data(), 32, MPI_UINT64_T, node.comm);
    prof.mark("wait");
    u64 arcs = 0;
    for (int q = 0; q < P; q++) arcs += owned[q];

    // The CSR: owner q's col, then w, in q's arena; an owner too heavy for its arena gets a window
    // of its own (every rank knows every owner's size, so all agree on it). The per-vertex
    // tables: one window, small.
    Graph g;
    g.arcs = arcs;
    g.relabeled = relabel;
    auto csr_bytes = [&](int q) { return page_up(4 * owned[q]) + owned[q]; };
    std::vector<char *> csr = arena;
    bool spill = false;
    for (int q = 0; q < P; q++) spill |= csr_bytes(q) > arena_bytes(q);
    const bool me = csr_bytes(r) > arena_bytes(r);
    if (spill) {
        const std::vector<char *> extra = node.alloc_each(me ? csr_bytes(r) : 0, "csr");
        for (int q = 0; q < P; q++)
            if (csr_bytes(q) > arena_bytes(q)) csr[q] = extra[q];
    }
    prof.count("csr-spills", me ? 1 : 0);
    g.col.resize(P);
    g.w.resize(P);
    for (int q = 0; q < P; q++) {
        g.col[q] = reinterpret_cast<u32 *>(csr[q]);
        g.w[q] = reinterpret_cast<u8 *>(csr[q] + page_up(4 * owned[q]));
    }
    char *const tables = node.alloc<char>(18 * N, "tables");
    g.adj = reinterpret_cast<Adj *>(tables);
    g.newid = reinterpret_cast<u32 *>(tables + 8 * N);
    g.oldid = reinterpret_cast<u32 *>(tables + 12 * N);
    g.own = reinterpret_cast<uint16_t *>(tables + 16 * N);

    // Slots. Relabeled: classes from the heaviest down, then owners in rank order, then positions;
    // my first slot in class c follows every heavier class and the lower ranks' share of c.
    u64 next[32] = {};
    if (relabel) {
        u64 acc = 0;
        for (int c = 31; c >= 0; c--) {
            next[c] = acc;
            for (int q = 0; q < P; q++) {
                if (q < r) next[c] += classes[u64(q) * 32 + c];
                acc += classes[u64(q) * 32 + c];
            }
        }
        g.n = acc;
    } else {
        g.n = N;
    }
    for (u64 li = 0, at = 0; li < nloc; li++) {
        const u32 v = mix.inv(u32(r + li * P)), d = pos[li];
        u32 s = v;
        if (relabel) {
            if (!d) {
                g.newid[v] = INF;
                continue;
            }
            s = u32(next[31 - __builtin_clz(d)]++);
            g.newid[v] = s;
            g.oldid[s] = v;
        }
        g.adj[s] = {u32(at), d};
        g.own[s] = uint16_t(r);
        pos[li] = u32(at);   // now the fill cursor, within my part of the CSR
        at += d;
    }
    prof.mark("slots");
    node.sync();   // every newid is in place before any owner translates its arcs' far ends
    prof.mark("wait");
#ifdef HW21_PROF
    // Experiments that split the fill's time. PP_PRETOUCH: the first touch of my CSR pages, done
    // in order beforehand. PP_PROBE: the fill's loop without its two stores, on cursors of its
    // own (copied first, under probe-copy): count's work plus the far ends' slots.
    if (std::getenv("PP_PRETOUCH")) {
        std::fill(g.col[r], g.col[r] + mine, 0u);
        std::fill(g.w[r], g.w[r] + mine, u8(0));
        prof.mark("pretouch");
    }
    if (std::getenv("PP_PROBE")) {
        std::vector<u32> cur(pos);
        prof.mark("probe-copy");
        u64 sum = 0;
        for_my_arcs([&](u64 a) { sum += cur[fd.div(u32(a & VMASK))]++ ^ g.slot(u32((a >> VBITS) & VMASK)); });
        volatile u64 keep = sum;   // the loop must run
        (void)keep;
        prof.mark("probe");
    }
#endif

    // Second pass fills them.
    u32 *const col = g.col[r];
    u8 *const wt = g.w[r];
    for_my_arcs([&](u64 a) {
        const u32 at = pos[fd.div(u32(a & VMASK))]++;
        col[at] = g.slot(u32((a >> VBITS) & VMASK));
        wt[at] = u8(a >> 52);
    });
    std::vector<u32>().swap(pos);
    prof.mark("fill");
    node.sync();
    prof.mark("wait");
#ifdef HW21_PROF
    if (std::getenv("PP_STATS")) {
        graph_stats(node, g, N, mix, nloc);
        prof.mark("stats");
    }
#endif
    return g;
}

// Atomic min on a window word last read as cur; true if this call lowered it.
inline bool lower(u32 *slot, u32 cur, u32 nd)
{
    while (nd < cur)
        if (__atomic_compare_exchange_n(slot, &cur, nd, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return true;
    return false;
}

// One batch of distinct sources, searched together over slots. Items are packed (s << 26 | slot).
class Search {
public:
    Search(Node &node, const Graph &g, int cap, Prof &prof)
        : node(node), g(g), n(g.n), prof(prof), P(node.size), r(node.rank)
    {
        // A slot is settled at most once per source, so n * cap bounds one round's frontier; the
        // coarse prefix keeps one entry per 64 items plus two per rank.
        dist = node.alloc<u32>(n * cap, "dist");
        front = node.alloc<u32>(n * cap, "front");
        coarse = node.alloc<u64>(n * cap / 64 + 2 * u64(P) + 1, "coarse");
        const char *bm = std::getenv("PP_BITMAP");
        bitmap = !(bm && bm[0] == '0');
        if (bitmap) final_bits = node.alloc<u64>((n * cap + 63) / 64, "bitmap");
        info.resize(3 * u64(P));
        off.resize(P);
        coff.resize(P);
        arc0.resize(P + 1);
    }

    // Distances from slots srcs[0..k) into dist[s * n + slot]; INF where unreachable.
    void run(const u32 *srcs, int k)
    {
        const u64 len = n * u64(k), lo = len * r / P, hi = len * (r + 1) / P;
        std::fill(dist + lo, dist + hi, INF);
        if (bitmap) {
            const u64 words = (len + 63) / 64;
            std::fill(final_bits + words * r / P, final_bits + words * (r + 1) / P, u64(0));
        }
        node.sync();
        for (int s = 0; s < k; s++)
            if (s % P == r) {
                dist[u64(s) * n + srcs[s]] = 0;
                ring[0].push_back(u32(s) << VBITS | srcs[s]);
            }

        // The profiling build splits the rounds' time: filter (finding and compacting the next
        // bucket), sync (the allgather and the barrier, waiting included), publish (writing the
        // frontier) and relax. A round with little work is mostly sync.
        double tf = 0, ts = 0, tp = 0, tr = 0;
        u64 small = 0, biggest = 0;
        u64 next = 0;   // every pending value lies in [next, next + 255]
        u64 rounds = 0, scanned = 0;
        for (;;) {
            const double t0 = tick();
            // My smallest pending value whose bucket still holds a valid item. Stale items
            // (lowered again since they were filed) are dropped on the way.
            u64 mine[3] = {NONE, 0, 0};
            for (u64 x = next; x < next + RING; x++) {
                std::vector<u32> &b = ring[x % RING];
                if (b.empty()) continue;
                size_t m = 0;
                u64 a = 0;
                for (const u32 it : b) {
                    const u32 v = it & VMASK;
                    if (__atomic_load_n(&dist[u64(it >> VBITS) * n + v], __ATOMIC_RELAXED) == x) {
                        b[m++] = it;
                        a += g.adj[v].deg;
                    }
                }
                b.resize(m);
                if (m) {
                    mine[0] = x;
                    mine[1] = m;
                    mine[2] = a;
                    break;
                }
            }
            const double t1 = tick();
            node.fence();
            MPI_Allgather(mine, 3, MPI_UINT64_T, info.data(), 3, MPI_UINT64_T, node.comm);
            node.fence();
            const double t2 = tick();
            tf += t1 - t0;
            ts += t2 - t1;
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
                // d is the smallest pending value anywhere, and a relaxation gives at least d, so
                // what is published is final: its bit goes up before the round's relaxations.
                std::vector<u32> &b = ring[d % RING];
                u64 a = 0;
                for (size_t j = 0; j < b.size(); j++) {
                    if (j % 64 == 0) coarse[coff[r] + j / 64] = a;
                    front[off[r] + j] = b[j];
                    a += g.adj[b[j] & VMASK].deg;
                    if (bitmap) {
                        const u64 i = u64(b[j] >> VBITS) * n + (b[j] & VMASK);
                        __atomic_fetch_or(final_bits + i / 64, u64(1) << (i % 64), __ATOMIC_RELAXED);
                    }
                }
                coarse[coff[r] + (b.size() + 63) / 64] = a;
                // Published, so the private copy can go; a big one goes for good, or 256 buckets
                // each keeping its peak capacity outgrow the private cap (scale 24 at P = 1 did).
                if (b.capacity() > (u64(1) << 16))
                    std::vector<u32>().swap(b);
                else
                    b.clear();
            }
            const double t3 = tick();
            node.sync();
            const double t4 = tick();

            const u64 T = arc0[P];
            relax_range(u32(d), T * r / P, T * (r + 1) / P);
            tp += t3 - t2;
            ts += t4 - t3;
            tr += tick() - t4;
            next = d + 1;
            rounds++;
            scanned += T * (r + 1) / P - T * r / P;
            if (Prof::on) {
                small += T < 1024 * u64(P);
                biggest = std::max(biggest, T);
            }
        }
        prof.sub("trav:", "filter", tf);
        prof.sub("trav:", "sync", ts);
        prof.sub("trav:", "publish", tp);
        prof.sub("trav:", "relax", tr);
        prof.count("rounds", r == 0 ? rounds : 0);
        prof.count("rounds-small", r == 0 ? small : 0);   // fewer than 1024 arcs per rank
        prof.count("round-max-arcs", r == 0 ? biggest : 0);
        prof.count("arcs", scanned);
        prof.count("pushes", pushes);
        prof.count("relax-final", final_hits);
        prof.count("relax-bitmap", bitmap_hits);
        prof.count("relax-nogain", nogain_hits);
        pushes = final_hits = bitmap_hits = nogain_hits = 0;
    }

    const u32 *distances() const { return dist; }

private:
    Node &node;
    const Graph &g;
    const u64 n;
    Prof &prof;
    const int P, r;
    u32 *dist, *front;
    u64 *coarse;
    u64 *final_bits = nullptr;   // bit s * n + slot: that pair's distance is final
    bool bitmap = false;
    u32 zero = 0;                // what a relaxation of a final target reads instead of its distance
    std::vector<u32> ring[RING];
    std::vector<u64> info, off, coff, arc0;
    // Relaxations: lowered, target final (caught by the bitmap), no lower.
    u64 pushes = 0, final_hits = 0, bitmap_hits = 0, nogain_hits = 0;

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
            if (bitmap)
                relax<true>(it >> VBITS, d, g.cols(it & VMASK) + o, g.weights(it & VMASK) + o, take);
            else
                relax<false>(it >> VBITS, d, g.cols(it & VMASK) + o, g.weights(it & VMASK) + o, take);
            left -= take;
            o = 0;
            i++;
        }
    }

    bool is_final(u64 i) const { return __atomic_load_n(final_bits + i / 64, __ATOMIC_RELAXED) >> (i % 64) & 1; }

    // Relaxes k arcs of source s from distance d. With the bitmap, a final target is passed over
    // without its distance being read: it reads zero, which lowers nothing. The bitmap's word is
    // fetched 2 PF arcs ahead and the distance PF ahead, only for a target not final, so the
    // distances of final targets never leave memory.
    template <bool BITMAP>
    void relax(u32 s, u32 d, const u32 *col, const u8 *w, u64 k)
    {
        u32 *ds = dist + u64(s) * n;
        const u64 base = u64(s) * n;
        constexpr u64 PF = 16;
        for (u64 e = 0; e < k; e++) {
            if (BITMAP) {
                if (e + 2 * PF < k) __builtin_prefetch(final_bits + (base + col[e + 2 * PF]) / 64);
                if (e + PF < k) {
                    const u32 z = col[e + PF];
                    __builtin_prefetch(is_final(base + z) ? &zero : ds + z);
                }
            } else if (e + PF < k) {
                __builtin_prefetch(ds + col[e + PF]);
            }
            const u32 y = col[e], nd = d + w[e];
            const bool fin = BITMAP && is_final(base + y);
            u32 *const at = fin ? &zero : ds + y;
            const u32 cur = __atomic_load_n(at, __ATOMIC_RELAXED);
            if (Prof::on) {   // how it ends, unless it lowers y (pushes) or loses a race (the rest)
                final_hits += cur <= d;
                bitmap_hits += fin;
                nogain_hits += cur > d && cur <= nd;
            }
            if (lower(at, cur, nd)) {
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
        Prof prof;
        Node node(comm, prof);
        const int P = node.size, r = node.rank;
        const FastDiv fd{u32(P)};
        const u64 N = num_vertices(p);

        const Graph g = build(node, p, fd, prof);
        const std::vector<u32> samples = r == 0 ? sample_vertices(p) : std::vector<u32>();

        // Distinct sources, first-occurrence order; repeats copy their twin's result. A source
        // with no slot is isolated: it reaches itself alone, and needs no search.
        std::vector<u32> uniq;
        std::vector<int> which(sources.size());
        for (size_t i = 0; i < sources.size(); i++) {
            const auto it = std::find(uniq.begin(), uniq.end(), sources[i]);
            which[i] = int(it - uniq.begin());
            if (it == uniq.end()) uniq.push_back(sources[i]);
        }
        std::vector<SourceResult> found(uniq.size());
        std::vector<u32> todo, todo_slot;   // the searched ones: index into uniq, and slot
        for (size_t u = 0; u < uniq.size(); u++) {
            const u32 s = g.slot(uniq[u]);
            if (s != INF) {
                todo.push_back(u32(u));
                todo_slot.push_back(s);
            } else if (r == 0) {
                SourceResult &res = found[u];
                res.n_reached = 1;
                res.checksum = 0;
                res.digest1 = distance_digest(uniq[u], 0, 0);
                res.digest2 = distance_digest(uniq[u], 0, 1);
                res.sample_dist.resize(samples.size());
                for (size_t k = 0; k < samples.size(); k++) res.sample_dist[k] = samples[k] == uniq[u] ? 0 : -1;
            }
        }

        // dist + frontier + coarse prefix + bitmap cost about 8.25 bytes per slot and source.
        const u64 budget = u64(20) << 30;
        int cap = int(std::max<u64>(1, std::min<u64>(todo.size(), budget / (g.n * 8 + g.n / 4 + 1))));
        if (const char *e = std::getenv("PP_BATCH")) cap = std::max(1, std::min(cap, std::atoi(e)));
        prof.count("batches", r == 0 ? (todo.size() + cap - 1) / cap : 0);

        Search search(node, g, cap, prof);
        prof.mark("setup");

        for (size_t b0 = 0; b0 < todo.size(); b0 += cap) {
            const int k = int(std::min<size_t>(cap, todo.size() - b0));
            search.run(todo_slot.data() + b0, k);
            prof.mark("traverse");

            // n_reached, checksum, digest1, digest2 over my slice of the slots.
            const u32 *dist = search.distances();
            std::vector<u64> sums(4 * u64(k), 0), total(4 * u64(k), 0);
            const u64 s0 = g.n * r / P, s1 = g.n * (r + 1) / P;
            for (int s = 0; s < k; s++) {
                const u32 *ds = dist + u64(s) * g.n;
                u64 cnt = 0, cs = 0, d1 = 0, d2 = 0;
                for (u64 x = s0; x < s1; x++) {
                    const u32 dv = ds[x];
                    if (dv == INF) continue;
                    const u32 v = g.vertex(u32(x));
                    cnt++;
                    cs += u64(dv) * (u64(v) + 1);
                    d1 += distance_digest(v, dv, 0);
                    d2 += distance_digest(v, dv, 1);
                }
                sums[4 * s] = cnt;
                sums[4 * s + 1] = cs;
                sums[4 * s + 2] = d1;
                sums[4 * s + 3] = d2;
            }
            MPI_Reduce(sums.data(), total.data(), 4 * k, MPI_UINT64_T, MPI_SUM, 0, comm);
            if (r == 0)
                for (int s = 0; s < k; s++) {
                    SourceResult &res = found[todo[b0 + s]];
                    res.n_reached = int64_t(total[4 * s]);
                    res.checksum = total[4 * s + 1];
                    res.digest1 = total[4 * s + 2];
                    res.digest2 = total[4 * s + 3];
                    const u32 *ds = dist + u64(s) * g.n;
                    res.sample_dist.resize(samples.size());
                    for (size_t j = 0; j < samples.size(); j++) {
                        const u32 x = g.slot(samples[j]);
                        res.sample_dist[j] = x == INF || ds[x] == INF ? -1 : int32_t(ds[x]);
                    }
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

        node.release_all();
        prof.mark("release");
        if (Prof::on && r == 0)
            std::fprintf(stderr,
                         "[hw2-1] P=%d N=%llu M=%llu arcs=%llu slots=%llu sources=%zu distinct=%zu searched=%zu batch=%d\n",
                         P, (unsigned long long)N, (unsigned long long)num_edges(p), (unsigned long long)g.arcs,
                         (unsigned long long)g.n, sources.size(), uniq.size(), todo.size(), cap);
        prof.report(comm, r);
    }
    MPI_Comm_free(&comm);
}
