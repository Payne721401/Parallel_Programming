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
PP_HD inline AxisTiles influence(int tile, int tile_size, int extent,
                                 int radius, bool torus) {
    AxisTiles result;
    const int begin = tile * tile_size;
    const int end = begin + tile_size < extent ? begin + tile_size : extent;
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
} // namespace tile_geometry

#undef PP_HD
