#pragma once

#include "tile_geometry.hpp"

// Geometry and queue candidates shared with the CPU orchestration test.
// Wider tiles trade finer support tracking for contiguous 32-byte packed rows.
#ifdef __CUDACC__
#define PP_SPARSE_WIDE_HD __host__ __device__
#else
#define PP_SPARSE_WIDE_HD
#endif

namespace r1_sparse_wide {
constexpr int width = 256;
constexpr int height = 16;
constexpr int words_per_row = width / 32;
constexpr int rows_per_lane = 4;
constexpr int warps_per_block = 4;
static_assert(32 * rows_per_lane == words_per_row * height, "one warp owns each output word once");
PP_SPARSE_WIDE_HD constexpr int column(int lane) { return lane % words_per_row; }
PP_SPARSE_WIDE_HD constexpr int first_row(int lane) {
    return lane / words_per_row * rows_per_lane;
}

struct Bounds {
    int xmin = width, ymin = height, xmax = -1, ymax = -1;
    PP_SPARSE_WIDE_HD void include(int x, int y) {
        xmin = x < xmin ? x : xmin;
        ymin = y < ymin ? y : ymin;
        xmax = x > xmax ? x : xmax;
        ymax = y > ymax ? y : ymax;
    }
    PP_SPARSE_WIDE_HD bool empty() const { return xmax < xmin; }
};

// Each participating lane owns at most one candidate, then stamp atomics
// select unique writers. The self candidate also preserves old-buffer cleanup.
PP_SPARSE_WIDE_HD inline int candidate(int lane, const tile_geometry::AxisTiles &xs,
                                      const tile_geometry::AxisTiles &ys,
                                      int tiles_x, int self) {
    const int neighbors = xs.count * ys.count;
    if (lane < neighbors)
        return ys.ids[lane / xs.count] * tiles_x + xs.ids[lane % xs.count];
    return lane == neighbors ? self : -1;
}
} // namespace r1_sparse_wide

#undef PP_SPARSE_WIDE_HD
