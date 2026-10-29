#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define PP_ROLL_HD __host__ __device__ __forceinline__
#else
#define PP_ROLL_HD inline
#endif

// Geometry and arithmetic shared by the CUDA implementation and its CPU tests.
namespace rolling_filters {
constexpr int dense_width = 128;
constexpr int dense_height = 64;
constexpr int line_rows = 16;
PP_ROLL_HD constexpr int align4(int n) { return (n + 3) & ~3; }
PP_ROLL_HD constexpr int align2(int n) { return (n + 1) & ~1; }
PP_ROLL_HD constexpr int halo_x(int radius) { return align4(radius); }
template <int Width, int Height>
PP_ROLL_HD constexpr int source_pitch(int radius) { return Width + 2 * halo_x(radius); }
template <int Width, int Height>
PP_ROLL_HD constexpr std::size_t source_bytes(int radius) {
    return std::size_t(source_pitch<Width, Height>(radius)) * (Height + 2 * radius);
}
template <int Width, int Height>
PP_ROLL_HD constexpr std::size_t shared_bytes(int radius, bool moore) {
    const int pitch = source_pitch<Width, Height>(radius);
    return source_bytes<Width, Height>(radius) + (moore
        ? std::size_t(pitch) * Height + std::size_t(8) * (pitch / 4 + 1) * sizeof(uint16_t)
        : std::size_t(align2(Width + radius)) * (Height + radius) * sizeof(uint16_t));
}
// The source byte lanes are 0/1. A vertical count is <=65, and each outgoing
// bit is already included in its count: subtraction cannot borrow across lanes.
PP_ROLL_HD uint32_t slide_bytes(uint32_t counts, uint32_t outgoing, uint32_t incoming) {
    return (counts - outgoing) + incoming;
}
PP_ROLL_HD unsigned byte_prefix(uint32_t values, int count) {
    unsigned sum = 0;
    for (int i = 0; i < count; ++i) sum += (values >> (8 * i)) & 255u;
    return sum;
}
// Uplus is the (R+1)-cell diagonal sum; Uminus omits its final endpoint.
PP_ROLL_HD uint16_t pack_diagonal(unsigned plus, unsigned last_alive) {
    return uint16_t(plus | ((plus - last_alive) << 8));
}
PP_ROLL_HD int diagonal_plus(uint16_t value) { return value & 255; }
PP_ROLL_HD int diagonal_minus(uint16_t value) { return value >> 8; }
} // namespace rolling_filters

#undef PP_ROLL_HD
