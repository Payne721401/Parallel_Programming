#pragma once

// Shared by CUDA kernels and a CPU-only exhaustive boundary test.
#ifdef __CUDACC__
#define PP_HD __host__ __device__
#else
#define PP_HD
#endif

namespace tile_geometry {
constexpr int width = 64;
constexpr int height = 32;

// Tile-local inclusive bounding box of actual state==1 cells. Empty is explicit.
struct Bounds {
    int xmin = width, ymin = height, xmax = -1, ymax = -1;
    PP_HD void include(int x, int y) {
        xmin = x < xmin ? x : xmin;
        ymin = y < ymin ? y : ymin;
        xmax = x > xmax ? x : xmax;
        ymax = y > ymax ? y : ymax;
    }
    PP_HD bool empty() const { return xmax < xmin; }
};

struct AxisTiles {
    int ids[4]{};
    int count = 0;

    PP_HD void append(int id) {
        for (int i = 0; i < count; ++i)
            if (ids[i] == id) return;
        ids[count++] = id;
    }
};

// All tiles intersecting the source tile's R-expanded *cell* interval.
// R<=32 and tile_size>=32 imply at most four distinct tile IDs, including
// a narrow final tile across a torus seam. Never wrap tile indices directly:
// wrapping cells and then dividing is different for partial final tiles.
// [begin,end) must lie within one valid source tile. Thus the same four-ID
// bound applies to both a complete tile and an alive bounding interval.
PP_HD inline AxisTiles influence_interval(int begin, int end, int tile_size, int extent,
                                          int radius, bool torus) {
    AxisTiles result;
    if (begin >= end) return result;
    int lo = begin - radius;
    int length = end - begin + 2 * radius;
    if (!torus) {
        const int hi = end + radius < extent ? end + radius : extent;
        if (lo < 0) lo = 0;
        for (int id = lo / tile_size; id <= (hi - 1) / tile_size; ++id)
            result.append(id);
    } else if (length >= extent) {
        for (int id = 0; id < (extent + tile_size - 1) / tile_size; ++id)
            result.append(id);
    } else {
        lo %= extent;
        if (lo < 0) lo += extent;
        // Consume a maximal segment within a tile, then wrap coordinates.
        while (length > 0) {
            const int id = lo / tile_size;
            result.append(id);
            const int edge = (id + 1) * tile_size < extent ? (id + 1) * tile_size : extent;
            const int take = length < edge - lo ? length : edge - lo;
            length -= take;
            lo += take;
            if (lo == extent) lo = 0;
        }
    }
    return result;
}

PP_HD inline AxisTiles influence(int tile, int tile_size, int extent,
                                 int radius, bool torus) {
    const int begin = tile * tile_size;
    const int end = begin + tile_size < extent ? begin + tile_size : extent;
    return influence_interval(begin, end, tile_size, extent, radius, torus);
}
} // namespace tile_geometry

#undef PP_HD
