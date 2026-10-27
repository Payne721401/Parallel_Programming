#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define PP_BITS_HD __host__ __device__ __forceinline__
#else
#define PP_BITS_HD inline
#endif

namespace r1_bits {
struct Count { uint32_t b0, b1, b2, b3; };
PP_BITS_HD void add3(uint32_t a, uint32_t b, uint32_t c, uint32_t &sum, uint32_t &carry) {
    sum = a ^ b ^ c;
    carry = (a & b) | ((a ^ b) & c);
}
// Each bit position is an INDEPENDENT count, never a popcount across cells.
PP_BITS_HD Count count9(uint32_t a, uint32_t b, uint32_t c,
                       uint32_t d, uint32_t e, uint32_t f,
                       uint32_t g, uint32_t h, uint32_t i) {
    uint32_t s0,c0,s1,c1,s2,c2,s3,c3,s4,c4;
    add3(a,b,c,s0,c0); add3(d,e,f,s1,c1); add3(g,h,i,s2,c2);
    add3(s0,s1,s2,s3,c3); add3(c0,c1,c2,s4,c4);
    const uint32_t carry4 = s4 & c3;
    return {s3, s4 ^ c3, c4 ^ carry4, c4 & carry4};
}
PP_BITS_HD uint32_t ge(Count n, unsigned threshold) {
    uint32_t eq = ~uint32_t(0), gt = 0;
    const uint32_t bits[4] = {n.b0,n.b1,n.b2,n.b3};
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int b=3; b>=0; --b) {
        if ((threshold >> b) & 1u) eq &= bits[b];
        else { gt |= eq & bits[b]; eq &= ~bits[b]; }
    }
    return gt | eq;
}
PP_BITS_HD uint32_t next(uint32_t state, Count n, unsigned slo, unsigned shi,
                        unsigned blo, unsigned bhi) {
    const uint32_t survive = ge(n,slo) & ~ge(n,shi+1);
    const uint32_t birth = ge(n,blo) & ~ge(n,bhi+1);
    return (state & survive) | (~state & birth);
}
PP_BITS_HD uint32_t mask(int width, int word) {
    const int valid = width - word*32;
    return valid >= 32 ? ~uint32_t(0) : valid <= 0 ? 0 : (uint32_t(1)<<valid)-1;
}
} // namespace r1_bits
#undef PP_BITS_HD
