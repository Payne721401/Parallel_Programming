#pragma once
#include <cstdint>
#include "ca.hpp"

#ifdef __CUDACC__
#define PP_R1_HD __host__ __device__ __forceinline__
#else
#define PP_R1_HD inline
#endif

namespace r1_packed {
constexpr uint32_t ones = 0x01010101u;
constexpr uint32_t high = 0x80808080u;

// Four independent byte lanes; states are 0..15, counts are 0..9.
// Setting each high bit prevents inter-byte borrow during subtraction.
PP_R1_HD uint32_t equal(uint32_t value, unsigned target) {
    const uint32_t x = value ^ (target * ones);
    return (~((x | high) - ones) >> 7) & ones;
}

PP_R1_HD uint32_t in_range(uint32_t counts, unsigned lo, unsigned hi) {
    const uint32_t ge = (counts | high) - lo * ones;
    const uint32_t le = ((hi * ones) | high) - counts;
    return ((ge & le) >> 7) & ones;
}

PP_R1_HD uint32_t left(uint32_t centre, uint32_t previous) {
    return (centre << 8) | (previous >> 24);
}

PP_R1_HD uint32_t right(uint32_t centre, uint32_t next) {
    return (centre >> 8) | (next << 24);
}

// Two adjacent output rows share the two middle input rows. Keeping their
// vertical sums in registers avoids loading those rows twice from shared memory.
struct CountPair { uint32_t first, second; };
template <bool Moore>
PP_R1_HD CountPair count_pair(uint32_t al, uint32_t a, uint32_t ar,
                              uint32_t bl, uint32_t b, uint32_t br,
                              uint32_t cl, uint32_t c, uint32_t cr,
                              uint32_t dl, uint32_t d, uint32_t dr, bool centre) {
    if constexpr (Moore) {
        const uint32_t l = bl + cl, m = b + c, r = br + cr;
        const uint32_t top = left(a + m, al + l) + (a + m) + right(a + m, ar + r);
        const uint32_t bottom = left(m + d, l + dl) + (m + d) + right(m + d, r + dr);
        return {centre ? top : top - b, centre ? bottom : bottom - c};
    } else {
        return {a + c + left(b, bl) + right(b, br) + (centre ? b : 0u),
                b + d + left(c, cl) + right(c, cr) + (centre ? c : 0u)};
    }
}

template <bool Moore>
PP_R1_HD uint32_t count(uint32_t ul, uint32_t u, uint32_t ur,
                        uint32_t ml, uint32_t m, uint32_t mr,
                        uint32_t dl, uint32_t d, uint32_t dr, bool centre) {
    // Inputs have one 0/1 alive indicator per byte. Every sum is <=9 per byte,
    // so ordinary uint32_t addition cannot carry into a neighbouring cell.
    if constexpr (Moore) {
        const uint32_t lsum = ul + ml + dl;
        const uint32_t mid = u + m + d;
        const uint32_t rsum = ur + mr + dr;
        const uint32_t all = left(mid, lsum) + mid + right(mid, rsum);
        return centre ? all : all - m; // every byte includes m, so no borrow
    } else {
        return u + d + left(m, ml) + right(m, mr) + (centre ? m : 0u);
    }
}

PP_R1_HD uint32_t next(uint32_t state, uint32_t counts, const Rule &r) {
    const uint32_t birth = in_range(counts, r.birth_lo, r.birth_hi);
    const uint32_t survive = in_range(counts, r.survive_lo, r.survive_hi);
    if (r.states == 2)
        return (state & survive) | ((state ^ ones) & birth);

    const uint32_t alive = equal(state, 1);
    const uint32_t dead = equal(state, 0);
    // Multiplication by 255 expands each 0/1 byte to 0/255 without carry.
    const uint32_t decay = ((alive | dead) ^ ones) * 255u;
    const uint32_t wrap = equal(state, unsigned(r.states - 1)) * 255u;
    return (dead & birth) | (alive & survive)
         | ((alive & (survive ^ ones)) << 1)
         | ((state + ones) & decay & ~wrap);
}
} // namespace r1_packed

#undef PP_R1_HD
