#pragma once
#include <cstddef>
#include <cstdint>
#include "ca.hpp"

#ifdef __CUDACC__
#define PP_MOORE_WARP_HD __host__ __device__ __forceinline__
#else
#define PP_MOORE_WARP_HD inline
#endif

// The CPU test uses the production ownership and horizontal-window arithmetic.
// It does not emulate CUDA shuffle execution, memory transactions or races.
namespace sparse_moore_warp {
constexpr int width = 64;
constexpr int height = 32;
constexpr int rows_per_warp = 16;
constexpr int strips_per_tile = height / rows_per_warp;
constexpr int warps_per_block = 4;
constexpr int threads_per_block = 32 * warps_per_block;
constexpr std::size_t shared_bytes = 0;

PP_MOORE_WARP_HD constexpr int halo_x(int radius) { return (radius + 3) & ~3; }
PP_MOORE_WARP_HD constexpr int source_pitch(int radius) { return width + 2 * halo_x(radius); }
PP_MOORE_WARP_HD constexpr int word_pitch(int radius) { return source_pitch(radius) / 4; }
PP_MOORE_WARP_HD constexpr bool eligible(const Problem &p) {
    return p.rule.shape == Shape::Moore && p.rule.radius >= 2 && p.rule.radius <= MAX_RADIUS;
}

PP_MOORE_WARP_HD unsigned byte_prefix(uint32_t word, int bytes) {
    unsigned sum = 0;
    for (int i = 0; i < bytes; ++i) sum += (word >> (8 * i)) & 255u;
    return sum;
}

// Four consecutive bytes, with no undefined shift by 32 at an aligned endpoint.
PP_MOORE_WARP_HD uint32_t align_bytes(uint32_t first, uint32_t second, int skip) {
    return skip ? (first >> (8 * skip)) | (second << (32 - 8 * skip)) : first;
}

struct Counts { unsigned cell[4]; };

// lo/hi are the [lo,hi) vertical-column endpoints for the first output cell.
// prefix_lo/prefix_hi exclude their containing word; packed columns are <=65.
// Horizontal totals may exceed255, so every final count is an ordinary integer.
PP_MOORE_WARP_HD Counts window4(unsigned prefix_lo, uint32_t lo0, uint32_t lo1, int lo_byte,
                               unsigned prefix_hi, uint32_t hi0, uint32_t hi1, int hi_byte) {
    Counts counts{};
    counts.cell[0] = prefix_hi + byte_prefix(hi0, hi_byte)
                   - prefix_lo - byte_prefix(lo0, lo_byte);
    const uint32_t outgoing = align_bytes(lo0, lo1, lo_byte);
    const uint32_t incoming = align_bytes(hi0, hi1, hi_byte);
    for (int b = 1; b < 4; ++b)
        counts.cell[b] = counts.cell[b - 1] - ((outgoing >> (8 * (b - 1))) & 255u)
                                           + ((incoming >> (8 * (b - 1))) & 255u);
    return counts;
}

// The removed bit is included in each byte count, so subtraction cannot borrow
// between byte lanes; incoming can raise each count only as far as65.
PP_MOORE_WARP_HD uint32_t slide(uint32_t counts, uint32_t outgoing, uint32_t incoming) {
    return counts - outgoing + incoming;
}
} // namespace sparse_moore_warp

#undef PP_MOORE_WARP_HD
