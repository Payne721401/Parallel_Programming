#pragma once

#include <cstddef>
#include <cstdint>
#include "ca.hpp"
#include "r1_bits.hpp"

#ifdef __CUDACC__
#define PP_MULTI_HD __host__ __device__ __forceinline__
#else
#define PP_MULTI_HD inline
#endif

// Binary bit planes: bit b of each cell's state is stored in plane b. Every
// uint32_t holds 32 independent cells. The compact C3 representation has no 11
// code, so its low plane is also its alive plane.
namespace r1_multistate {

PP_MULTI_HD int plane_count(int states) {
    return states <= 2 ? 1 : states <= 4 ? 2 : states <= 8 ? 3 : 4;
}

inline std::size_t plane_words(const Problem &p) {
    return std::size_t((p.width + 31) / 32) * std::size_t(p.height);
}

inline std::size_t allocation_bytes(const Problem &p) {
    const std::size_t bytes = std::size_t(p.width) * std::size_t(p.height);
    const std::size_t packed = plane_words(p) * std::size_t(plane_count(p.rule.states))
                            * sizeof(uint32_t);
    // Reuse a byte buffer for the packed representation, including W=3 tails.
    return packed > bytes ? packed : bytes;
}

template <int Planes> struct Word {
    static_assert(Planes >= 1 && Planes <= 4, "C2..16 bit-plane representation");
    uint32_t bit[Planes]{};
};

template <int Planes>
PP_MULTI_HD uint32_t equal(const Word<Planes> &state, unsigned value) {
    uint32_t result = ~uint32_t(0);
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int b = 0; b < Planes; ++b)
        result &= ((value >> b) & 1u) ? state.bit[b] : ~state.bit[b];
    return result;
}

template <int Planes>
PP_MULTI_HD uint32_t alive(const Word<Planes> &state, int states) {
    if constexpr (Planes == 2) {
        // This relies only on the legal C3 state invariant, maintained by next.
        if (states == 3) return state.bit[0];
    }
    return equal(state, 1);
}

template <int Planes>
PP_MULTI_HD unsigned extract(const Word<Planes> &state, unsigned lane) {
    unsigned result = 0;
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int b = 0; b < Planes; ++b)
        result |= ((state.bit[b] >> lane) & 1u) << b;
    return result;
}

struct Horizontal { uint32_t left, centre, right; };

// prev/next must be the wrapped word at the two torus seams. The partial last
// word wraps at the true width, rather than at the padded word pitch.
PP_MULTI_HD Horizontal horizontal(uint32_t centre, uint32_t prev, uint32_t next,
                                 int word, int width, bool torus) {
    uint32_t left = (centre << 1) | (prev >> 31);
    uint32_t right = (centre >> 1) | (next << 31);
    if (torus) {
        if (word == 0) left = (centre << 1) | ((prev >> ((width - 1) & 31)) & 1u);
        if (word == (width - 1) / 32)
            right = (centre >> 1) | ((next & 1u) << ((width - 1) & 31));
    }
    return {left, centre, right};
}

struct RuleMasks { uint32_t survive, birth; };

PP_MULTI_HD RuleMasks rule_masks(r1_bits::Count n, const Rule &r) {
    // A rule dispatch, independent of grid shape, case IDs or initial state.
    // B3/S23 needs no four generic four-bit threshold comparators.
    if (r.survive_lo == 2 && r.survive_hi == 3 &&
        r.birth_lo == 3 && r.birth_hi == 3) {
        const uint32_t two_or_three = n.b1 & ~(n.b2 | n.b3);
        return {two_or_three, two_or_three & n.b0};
    }
    return {
        r1_bits::ge_compact(n, unsigned(r.survive_lo)) &
            ~r1_bits::ge_compact(n, unsigned(r.survive_hi + 1)),
        r1_bits::ge_compact(n, unsigned(r.birth_lo)) &
            ~r1_bits::ge_compact(n, unsigned(r.birth_hi + 1))
    };
}

template <int Planes>
PP_MULTI_HD Word<Planes> next(const Word<Planes> &state, r1_bits::Count n,
                             const Rule &r) {
    const RuleMasks rule = rule_masks(n, r);
    Word<Planes> result;
    if constexpr (Planes == 1) {
        result.bit[0] = (state.bit[0] & rule.survive) |
                        (~state.bit[0] & rule.birth);
        return result;
    }
    if constexpr (Planes == 2) {
        const uint32_t a = alive(state, r.states);
        const uint32_t dead = ~(state.bit[0] | state.bit[1]);
        const uint32_t born_or_stay = (dead & rule.birth) | (a & rule.survive);
        const uint32_t fail = a & ~rule.survive;
        if (r.states == 3) {
            result.bit[0] = born_or_stay;
            result.bit[1] = fail;
        } else {
            const uint32_t state_two = state.bit[1] & ~state.bit[0];
            result.bit[0] = born_or_stay | state_two;
            result.bit[1] = fail | state_two;
        }
        return result;
    }
    const uint32_t a = equal(state, 1);
    const uint32_t dead = equal(state, 0);
    const uint32_t decay = ~(a | dead) & ~equal(state, unsigned(r.states - 1));
    const uint32_t fail = a & ~rule.survive;
    const uint32_t born_or_stay = (dead & rule.birth) | (a & rule.survive);
    uint32_t carry = ~uint32_t(0);
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int b = 0; b < Planes; ++b) {
        const uint32_t incremented = state.bit[b] ^ carry;
        carry &= state.bit[b];
        result.bit[b] = incremented & decay;
        if (b == 0) result.bit[b] |= born_or_stay;
        if (b == 1) result.bit[b] |= fail;
    }
    return result;
}

} // namespace r1_multistate

#undef PP_MULTI_HD
