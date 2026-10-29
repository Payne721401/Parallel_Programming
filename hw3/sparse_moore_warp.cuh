#pragma once

#ifdef __CUDACC__
// Include inside the implementation namespace, after r1_sparse_wide.cuh and
// the established TileQueue/AliveBounds definitions. Requires next_state,
// wrap_halo, r1_packed.hpp and sparse_moore_warp.hpp. No CTA collective or shared
// memory is used. Four complete warps form a 128-thread one-dimensional block.
namespace sparse_moore_warp_gpu {
static_assert(TILE_W == sparse_moore_warp::width && TILE_H == sparse_moore_warp::height,
              "Moore warp ownership retains the existing64x32 queue");

template <bool Torus>
__device__ __forceinline__ uint8_t source_cell(const uint8_t *src, const Problem &p, int x, int y) {
    if constexpr (Torus) { x = wrap_halo(x, p.width); y = wrap_halo(y, p.height); }
    if (x < 0 || x >= p.width || y < 0 || y >= p.height) return 0;
    return src[std::size_t(y) * p.width + x];
}

template <bool Torus>
__device__ __forceinline__ uint32_t source_word(const uint8_t *src, const Problem &p, int x, int y) {
    if constexpr (Torus) y = wrap_halo(y, p.height);
    if (y < 0 || y >= p.height) return 0;
    int gx = x;
    if constexpr (Torus) gx = wrap_halo(gx, p.width);
    if (gx >= 0 && gx + 3 < p.width) {
        const std::size_t index = std::size_t(y) * p.width + gx;
        if ((index & 3) == 0) return *reinterpret_cast<const uint32_t *>(src + index);
    }
    uint32_t value = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) value |= uint32_t(source_cell<Torus>(src, p, x + b, y)) << (8 * b);
    return value;
}

__device__ __forceinline__ unsigned sum_bytes(uint32_t counts) {
    // Each byte is <=65, including atR32, so signed dp4a is exact.
    return unsigned(__dp4a(int(counts), int(0x01010101u), 0));
}

template <bool Torus, bool Tight>
__global__ void step(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, Problem p,
                     TileQueue current, TileQueue next, int *stamp, int epoch,
                     const int *dense, int tiles_x, int tile_count,
                     const int *device_epoch = nullptr) {
    if (device_epoch) epoch = *device_epoch;
    constexpr unsigned mask = 0xffffffffu;
    const int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) / 32;
    const int radius = p.rule.radius, left = sparse_moore_warp::halo_x(radius);
    const int words = sparse_moore_warp::word_pitch(radius);
    const bool full = *dense != 0;
    const int count = full ? tile_count : *current.count;
    // Strip ownership doubles available work warps for small active frontiers.
    // The strips independently publish support; their union covers the tile.
    for (int work = int(blockIdx.x) * sparse_moore_warp::warps_per_block + warp;
         work < count * sparse_moore_warp::strips_per_tile;
         work += int(gridDim.x) * sparse_moore_warp::warps_per_block) {
        const int item = work / sparse_moore_warp::strips_per_tile;
        const int strip = work % sparse_moore_warp::strips_per_tile;
        const int tile = full ? item : current.ids[item];
        const int x0 = tile % tiles_x * TILE_W, y0 = tile / tiles_x * TILE_H;
        const int first_y = strip * sparse_moore_warp::rows_per_warp;
        if (y0 + first_y >= p.height) continue;
        const int source_x = x0 + 4 * lane - left;
        uint32_t vertical = 0;
        if (lane < words)
            for (int dy = -radius; dy <= radius; ++dy)
                vertical += r1_packed::equal(source_word<Torus>(src, p, source_x, y0 + first_y + dy), 1);
        AliveBounds bounds;
        unsigned flags = 0;
        for (int row = 0; row < sparse_moore_warp::rows_per_warp; ++row) {
            const int y = first_y + row, gy = y0 + y;
            const unsigned word_sum = sum_bytes(vertical);
            unsigned inclusive = word_sum;
#pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                const unsigned preceding = __shfl_up_sync(mask, inclusive, offset);
                if (lane >= offset) inclusive += preceding;
            }
            const unsigned exclusive = inclusive - word_sum;
            // All lanes shuffle, including the lanes without an output word.
            // Duplicating the lower-half requests keeps every source index in
            // [0,31], including theR32 endpoint at the final padded column.
            const int output_word = lane & 15;
            const int lo = left + output_word * 4 - radius;
            const int hi = left + output_word * 4 + radius + 1;
            const int lw = lo / 4, hw = hi / 4;
            const unsigned prefix_lo = __shfl_sync(mask, exclusive, lw);
            const unsigned prefix_hi = __shfl_sync(mask, exclusive, hw);
            const uint32_t lo0 = __shfl_sync(mask, vertical, lw);
            const uint32_t lo1_raw = __shfl_sync(mask, vertical, lw + 1 < 32 ? lw + 1 : 31);
            const uint32_t hi0 = __shfl_sync(mask, vertical, hw);
            const uint32_t hi1_raw = __shfl_sync(mask, vertical, hw + 1 < 32 ? hw + 1 : 31);
            const uint32_t lo1 = lw + 1 < words ? lo1_raw : 0;
            const uint32_t hi1 = hw + 1 < words ? hi1_raw : 0;
            if (lane < TILE_W / 4 && gy < p.height) {
                const auto counts = sparse_moore_warp::window4(prefix_lo, lo0, lo1, lo & 3,
                                                              prefix_hi, hi0, hi1, hi & 3);
                const int gx = x0 + lane * 4;
                const uint32_t state_word = source_word<Torus>(src, p, gx, gy);
                uint32_t output_word_value = 0;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint8_t state = uint8_t(state_word >> (8 * b));
                    const int n = int(counts.cell[b]) - int(!p.rule.count_centre && state == 1);
                    const uint8_t output = next_state(p.rule, state, n);
                    output_word_value |= uint32_t(output) << (8 * b);
                    if (gx + b < p.width && !full) {
                        flags |= unsigned(state != 0) | (unsigned(output != 0) << 1);
                        if constexpr (Tight) if (output == 1) bounds.include(lane * 4 + b, y);
                    }
                }
                if (gx < p.width) {
                    const std::size_t index = std::size_t(gy) * p.width + gx;
                    if ((index & 3) == 0 && gx + 3 < p.width)
                        *reinterpret_cast<uint32_t *>(dst + index) = output_word_value;
                    else {
#pragma unroll
                        for (int b = 0; b < 4; ++b)
                            if (gx + b < p.width) dst[index + b] = uint8_t(output_word_value >> (8 * b));
                    }
                }
            }
            if (row + 1 < sparse_moore_warp::rows_per_warp && lane < words) {
                const uint32_t outgoing = r1_packed::equal(source_word<Torus>(src, p, source_x, gy - radius), 1);
                const uint32_t incoming = r1_packed::equal(source_word<Torus>(src, p, source_x, gy + radius + 1), 1);
                vertical = sparse_moore_warp::slide(vertical, outgoing, incoming);
            }
        }
        if (!full) {
            flags = __reduce_or_sync(mask, flags);
            tile_geometry::AxisTiles xs, ys;
            if constexpr (Tight) {
                bounds.xmin = __reduce_min_sync(mask, bounds.xmin);
                bounds.ymin = __reduce_min_sync(mask, bounds.ymin);
                bounds.xmax = __reduce_max_sync(mask, bounds.xmax);
                bounds.ymax = __reduce_max_sync(mask, bounds.ymax);
                if (!bounds.empty()) {
                    xs = tile_geometry::influence_interval(x0 + bounds.xmin, x0 + bounds.xmax + 1,
                        TILE_W, p.width, radius, Torus);
                    ys = tile_geometry::influence_interval(y0 + bounds.ymin, y0 + bounds.ymax + 1,
                        TILE_H, p.height, radius, Torus);
                }
            } else if (flags & 2) {
                xs = tile_geometry::influence(tile % tiles_x, TILE_W, p.width, radius, Torus);
                ys = tile_geometry::influence(tile / tiles_x, TILE_H, p.height, radius, Torus);
            }
            // Old input must be cleared on destination reuse. New decay needs
            // its self tile even if there are no alive cells to dilate.
            const int self = (flags & (Tight ? 3u : 1u)) ? tile : -1;
            r1_sparse_wide_gpu::enqueue(xs, ys, self, next, stamp, epoch, tiles_x);
        }
    }
}

template <bool Tight>
void launch(const uint8_t *src, uint8_t *dst, const Problem &p,
            TileQueue current, TileQueue next, int *stamp, int epoch,
            const int *dense, int tiles_x, int tile_count, int workers,
            const int *device_epoch = nullptr, cudaStream_t stream = nullptr) {
    // The established workers budget is for 256-thread CTAs. Double the CTA
    // count here to retain its available warp budget, rather than accidentally
    // cutting resident work to one quarter when mapping two strips per tile.
    // Passing workers>=1 is the same precondition as for the established path.
    const int max_blocks = (tile_count * sparse_moore_warp::strips_per_tile
                            + sparse_moore_warp::warps_per_block - 1)
                          / sparse_moore_warp::warps_per_block;
    const int blocks = workers * 2 < max_blocks ? workers * 2 : max_blocks;
    if (p.boundary == Boundary::Torus)
        step<true, Tight><<<blocks, sparse_moore_warp::threads_per_block, 0, stream>>>(
            src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count, device_epoch);
    else
        step<false, Tight><<<blocks, sparse_moore_warp::threads_per_block, 0, stream>>>(
            src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count, device_epoch);
}
} // namespace sparse_moore_warp_gpu
#endif // __CUDACC__
