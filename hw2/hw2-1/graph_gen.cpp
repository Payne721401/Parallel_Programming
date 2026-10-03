// HW 2-1: the graph generator. BASE FILE: see graph_gen.hpp.
#include "graph_gen.hpp"

#include <algorithm>
#include <unordered_set>
#include <cstddef>

namespace {

// splitmix64's finalizer. Every random decision here is a hash of (seed, index, ...), so edge i
// costs only its own arithmetic.
inline uint64_t mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline uint64_t hash2(uint64_t a, uint64_t b) { return mix(mix(a) ^ (b * 0xd6e8feb86659fd93ULL)); }

// A bijection on [0, 2^scale): three multiplications by odd constants (bijective modulo
// 2^scale) interleaved with xor-shifts (bijective for any shift >= 1), keyed by the seed.
inline uint32_t scramble_id(uint64_t seed, int scale, uint32_t v)
{
    const uint64_t mask = (uint64_t(1) << scale) - 1;
    const int sh = scale / 2 > 0 ? scale / 2 : 1;
    const uint64_t k1 = mix(seed ^ 0x1111ULL) | 1, k2 = mix(seed ^ 0x2222ULL) | 1, k3 = mix(seed ^ 0x3333ULL) | 1;
    uint64_t x = v;
    x = (x * k1) & mask;
    x ^= x >> sh;
    x = (x * k2) & mask;
    x ^= x >> sh;
    x = (x * k3) & mask;
    return (uint32_t)x;
}

}  // namespace

void generate_edge(const GraphParams &p, uint64_t i, uint32_t &u, uint32_t &v, uint8_t &w)
{
    // Kronecker descent: at every level one byte of hash decides the quadrant. The thresholds
    // are Graph500's A = 0.57, B = 0.19, C = 0.19, D = 0.05 at 8-bit resolution:
    // [0,146) -> (0,0), [146,195) -> (0,1), [195,243) -> (1,0), [243,256) -> (1,1).
    // One 64-bit hash covers eight levels.
    uint64_t state = hash2(p.seed, i), bits = state;
    uint32_t a = 0, b = 0;
    for (int level = 0; level < p.scale; level++) {
        if (level && (level & 7) == 0) { state = mix(state); bits = state; }
        const unsigned r = (unsigned)(bits & 0xff);
        bits >>= 8;
        const uint32_t bit_a = r >= 195, bit_b = (r >= 146 && r < 195) || r >= 243;
        a |= bit_a << level;
        b |= bit_b << level;
    }
    if (p.scramble) {
        a = scramble_id(p.seed ^ 0x5c2a3b1eULL, p.scale, a);
        b = scramble_id(p.seed ^ 0x5c2a3b1eULL, p.scale, b);
    }
    u = a;
    v = b;
    // the weight belongs to the unordered pair, so the same edge generated twice weighs the same
    const uint64_t lo = std::min(a, b), hi = std::max(a, b);
    w = (uint8_t)(1 + hash2(p.seed ^ 0x77e19470ULL, (hi << 32) | lo) % 255);
}

void generate_edges(const GraphParams &p, uint64_t i0, uint64_t i1, uint32_t *u, uint32_t *v, uint8_t *w)
{
    for (uint64_t i = i0; i < i1; i++) generate_edge(p, i, u[i - i0], v[i - i0], w[i - i0]);
}

std::vector<uint32_t> sample_vertices(const GraphParams &p)
{
    const uint64_t n = num_vertices(p);
    std::vector<uint32_t> s;
    s.reserve(p.nsample);
    if (uint64_t(p.nsample) == n) {
        for (uint32_t v = 0; v < n; ++v) s.push_back(v);
        return s;
    }
    std::unordered_set<uint32_t> seen;
    seen.reserve(p.nsample);
    uint64_t k = 0;
    while ((int)s.size() < p.nsample && (int)s.size() < (int)std::min<uint64_t>(n, 1u << 30)) {
        uint32_t v = (uint32_t)(hash2(p.seed ^ 0x5a4d9e11ULL, k++) % n);
        if (seen.insert(v).second) s.push_back(v);
    }
    std::sort(s.begin(), s.end());
    return s;
}
