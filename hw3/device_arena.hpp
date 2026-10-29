#pragma once

#include <cstddef>
#include <limits>

// CPU-only address layout shared by the CUDA allocator and its tests. This
// helper neither allocates memory nor owns the pointers bound to its regions.
namespace device_arena {

constexpr std::size_t default_alignment = 256;
constexpr std::size_t absent_offset = std::numeric_limits<std::size_t>::max();

struct Region {
    std::size_t offset = absent_offset;
    std::size_t bytes = 0;
    bool present() const noexcept { return offset != absent_offset; }
};

struct Layout {
    Region src, dst, nonzero, ids[2], counts, stamp, stats;
    std::size_t total_bytes = 0;
    std::size_t alignment = default_alignment;
};

inline bool checked_add(std::size_t a, std::size_t b,
                        std::size_t &out) noexcept {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    out = a + b;
    return true;
}

inline bool checked_multiply(std::size_t a, std::size_t b,
                             std::size_t &out) noexcept {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) return false;
    out = a * b;
    return true;
}

inline bool valid_alignment(std::size_t alignment) noexcept {
    return alignment >= default_alignment && (alignment & (alignment - 1)) == 0;
}

inline bool checked_align_up(std::size_t value, std::size_t alignment,
                             std::size_t &out) noexcept {
    if (!valid_alignment(alignment)) return false;
    const std::size_t remainder = value & (alignment - 1);
    if (!remainder) {
        out = value;
        return true;
    }
    return checked_add(value, alignment - remainder, out);
}

// allocation_bytes is the *existing* per-grid allocation capacity, including
// any extra capacity needed by narrow bit-plane rows. It must not be replaced
// with width*height or just the packed payload. Dense layouts have no metadata;
// sparse tile counts must fit the int-based CUDA queue implementation. A zero
// stats_bytes omits optional profiling statistics without changing other data.
// On failure, out is reset and no partial layout can be used by the allocator.
inline bool make_layout(std::size_t allocation_bytes, bool sparse,
                        std::size_t tile_count, std::size_t stats_bytes,
                        Layout &out,
                        std::size_t alignment = default_alignment) noexcept {
    out = Layout{};
    if (!allocation_bytes || !valid_alignment(alignment)) return false;
    if (sparse) {
        if (!tile_count || tile_count > std::size_t(std::numeric_limits<int>::max()))
            return false;
    } else if (tile_count || stats_bytes) {
        return false;
    }

    Layout result;
    result.alignment = alignment;
    std::size_t cursor = 0;
    const auto append = [&](Region &region, std::size_t bytes) {
        std::size_t offset = 0, end = 0;
        if (!checked_align_up(cursor, alignment, offset) ||
            !checked_add(offset, bytes, end)) return false;
        region = {offset, bytes};
        cursor = end;
        return true;
    };
    if (!append(result.src, allocation_bytes) ||
        !append(result.dst, allocation_bytes)) return false;
    if (sparse) {
        std::size_t integer_bytes = 0;
        if (!checked_multiply(tile_count, sizeof(int), integer_bytes) ||
            !append(result.nonzero, tile_count) ||
            !append(result.ids[0], integer_bytes) ||
            !append(result.ids[1], integer_bytes) ||
            !append(result.counts, 4 * sizeof(int)) ||
            !append(result.stamp, integer_bytes)) return false;
        if (stats_bytes && !append(result.stats, stats_bytes)) return false;
    }
    if (!checked_align_up(cursor, alignment, result.total_bytes)) return false;
    out = result;
    return true;
}

} // namespace device_arena
