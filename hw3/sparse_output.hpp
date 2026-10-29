#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define PP_SPARSE_OUTPUT_HD __host__ __device__ __forceinline__
#else
#define PP_SPARSE_OUTPUT_HD inline
#endif

namespace sparse_output {

// This geometry must describe both initial_nonzero and the final queue. It is
// independent of the generation kernel, including its packed representation.
struct Geometry {
    int width = 0, height = 0;
    int tile_width = 0, tile_height = 0;
    int tiles_x = 0, tiles_y = 0, tile_count = 0;
};

PP_SPARSE_OUTPUT_HD Geometry geometry(int width, int height, int tile_width, int tile_height) {
    Geometry g;
    if (width <= 0 || height <= 0 || tile_width <= 0 || tile_height <= 0) return g;
    g.width = width; g.height = height;
    g.tile_width = tile_width; g.tile_height = tile_height;
    g.tiles_x = (width - 1) / tile_width + 1;
    g.tiles_y = (height - 1) / tile_height + 1;
    g.tile_count = g.tiles_x * g.tiles_y;
    return g;
}

PP_SPARSE_OUTPUT_HD bool valid(const Geometry &g) {
    return g.width > 0 && g.height > 0 && g.tile_width > 0 && g.tile_height > 0
        && g.tiles_x == (g.width - 1) / g.tile_width + 1
        && g.tiles_y == (g.height - 1) / g.tile_height + 1
        && g.tile_count == g.tiles_x * g.tiles_y;
}

// All products involving a grid offset use size_t. In particular, a 32768^2
// grid and offsets within it must not be formed by narrower intermediate math.
PP_SPARSE_OUTPUT_HD bool cell_index(const Geometry &g, int tile, int cell, std::size_t &index) {
    if (tile < 0 || tile >= g.tile_count || cell < 0
        || cell >= g.tile_width * g.tile_height) return false;
    const int x = (tile % g.tiles_x) * g.tile_width + cell % g.tile_width;
    const int y = (tile / g.tiles_x) * g.tile_height + cell / g.tile_width;
    if (x >= g.width || y >= g.height) return false;
    index = std::size_t(y) * std::size_t(g.width) + std::size_t(x);
    return true;
}

PP_SPARSE_OUTPUT_HD std::size_t tile_bytes(const Geometry &g, int tile) {
    if (tile < 0 || tile >= g.tile_count) return 0;
    const int x = (tile % g.tiles_x) * g.tile_width;
    const int y = (tile / g.tiles_x) * g.tile_height;
    const int width = g.width - x < g.tile_width ? g.width - x : g.tile_width;
    const int height = g.height - y < g.tile_height ? g.height - y : g.tile_height;
    return std::size_t(width) * std::size_t(height);
}

} // namespace sparse_output

#undef PP_SPARSE_OUTPUT_HD
