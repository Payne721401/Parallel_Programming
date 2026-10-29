#pragma once

// The CPU orchestration test shares this exact ownership map. It cannot
// validate CUDA shuffle execution or memory races.
#ifdef __CUDACC__
#define PP_SPARSE_BITS_HD __host__ __device__
#else
#define PP_SPARSE_BITS_HD
#endif
namespace r1_sparse_bits_layout {
constexpr int warps_per_block = 4;
constexpr int rows_per_lane = 2;
PP_SPARSE_BITS_HD constexpr int column(int lane) { return lane & 1; }
PP_SPARSE_BITS_HD constexpr int first_row(int lane) { return (lane / 2) * rows_per_lane; }
} // namespace r1_sparse_bits_layout
#undef PP_SPARSE_BITS_HD

#ifdef __CUDACC__
// Included after TileQueue/enqueue helpers. Keep the established 64x32 queue
// geometry and its cleanup invariant; four independent warps process four
// queued tiles per block. No CTA-wide collective is used in this kernel.
namespace r1_sparse_bits_gpu {
static_assert(TILE_W == 64 && TILE_H == 32, "sparse bit ownership requires 64x32 queue tiles");

template <int Planes, bool Torus>
__device__ __forceinline__ r1_multistate_gpu::Row<Planes>
load_row(const uint32_t *src, int wx, int y, const Problem &p) {
    const int column = r1_sparse_bits_layout::column(int(threadIdx.x) & 31);
    const int pitch = (p.width + 31) / 32;
    const auto state = r1_multistate_gpu::load_word<Planes, Torus>(src, wx, y, pitch, p.height);
    const uint32_t a = r1_multistate::alive(state, p.rule.states);
    uint32_t previous = __shfl_up_sync(0xffffffffu, a, 1, 2);
    uint32_t following = __shfl_down_sync(0xffffffffu, a, 1, 2);
    if (column == 0) {
        const int x = Torus && wx == 0 ? pitch - 1 : wx - 1;
        previous = r1_multistate::alive(
            r1_multistate_gpu::load_word<Planes, Torus>(src, x, y, pitch, p.height), p.rule.states);
    }
    if (column == 1 || wx == pitch - 1) {
        const int x = Torus && wx == pitch - 1 ? 0 : wx + 1;
        following = r1_multistate::alive(
            r1_multistate_gpu::load_word<Planes, Torus>(src, x, y, pitch, p.height), p.rule.states);
    }
    const auto shifted = r1_multistate::horizontal(a, previous, following, wx, p.width, Torus);
    return {state, shifted.left, shifted.centre, shifted.right};
}

template <int Planes, bool Moore, bool Torus, bool Tight>
__global__ void step(const uint32_t *src, uint32_t *dst, Problem p,
                     TileQueue current, TileQueue next, int *stamp, int epoch,
                     const int *dense, int tiles_x, int tile_count) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) / 32;
    const bool full = *dense != 0;
    const int count = full ? tile_count : *current.count;
    const int pitch = (p.width + 31) / 32;
    const size_t words = size_t(pitch) * p.height;
    for (int work = int(blockIdx.x) * r1_sparse_bits_layout::warps_per_block + warp;
         work < count; work += int(gridDim.x) * r1_sparse_bits_layout::warps_per_block) {
        const int tile = full ? work : current.ids[work];
        const int x0 = tile % tiles_x * TILE_W, y0 = tile / tiles_x * TILE_H;
        const int wx = x0 / 32 + r1_sparse_bits_layout::column(lane);
        const int y_first = y0 + r1_sparse_bits_layout::first_row(lane);
        auto upper = load_row<Planes, Torus>(src, wx, y_first - 1, p);
        auto middle = load_row<Planes, Torus>(src, wx, y_first, p);
        auto lower = load_row<Planes, Torus>(src, wx, y_first + 1, p);
        AliveBounds bounds;
        unsigned flags = 0;
#pragma unroll
        for (int row = 0; row < r1_sparse_bits_layout::rows_per_lane; ++row) {
            const int y = y_first + row;
            r1_bits::Count n;
            if constexpr (Moore)
                n = r1_bits::count9(upper.left, upper.centre, upper.right,
                    middle.left, middle.right, lower.left, lower.centre, lower.right,
                    p.rule.count_centre ? middle.centre : 0);
            else
                n = r1_bits::count9(0, upper.centre, 0, middle.left, middle.right,
                    0, lower.centre, 0, p.rule.count_centre ? middle.centre : 0);
            const auto output = r1_multistate::next(middle.state, n, p.rule);
            if (wx < pitch && y < p.height) {
                const uint32_t mask = r1_bits::mask(p.width, wx);
                const size_t index = size_t(y) * pitch + wx;
                uint32_t old_nonzero = 0, new_nonzero = 0;
#pragma unroll
                for (int b = 0; b < Planes; ++b) {
                    dst[size_t(b) * words + index] = output.bit[b] & mask;
                    old_nonzero |= middle.state.bit[b];
                    new_nonzero |= output.bit[b];
                }
                if (!full) {
                    flags |= unsigned((old_nonzero & mask) != 0)
                           | (unsigned((new_nonzero & mask) != 0) << 1);
                    if constexpr (Tight) {
                        const uint32_t alive = r1_multistate::alive(output, p.rule.states) & mask;
                        if (alive) {
                            bounds.include(wx * 32 - x0 + __ffs(alive) - 1, y - y0);
                            bounds.include(wx * 32 - x0 + 31 - __clz(alive), y - y0);
                        }
                    }
                }
            }
            if (row == 0) {
                upper = middle; middle = lower;
                lower = load_row<Planes, Torus>(src, wx, y_first + 2, p);
            }
        }
        if (!full) {
            flags = __reduce_or_sync(0xffffffffu, flags);
            if constexpr (Tight) {
                bounds.xmin = __reduce_min_sync(0xffffffffu, bounds.xmin);
                bounds.ymin = __reduce_min_sync(0xffffffffu, bounds.ymin);
                bounds.xmax = __reduce_max_sync(0xffffffffu, bounds.xmax);
                bounds.ymax = __reduce_max_sync(0xffffffffu, bounds.ymax);
            }
            if (lane == 0) {
                if constexpr (Tight) {
                    enqueue_alive_bounds(bounds, x0, y0, next, stamp, epoch, p, tiles_x);
                    if (flags & 3) enqueue_tile(tile, next, stamp, epoch);
                } else {
                    if (flags & 2) enqueue_influence(tile, next, stamp, epoch, p, tiles_x);
                    if (flags & 1) enqueue_tile(tile, next, stamp, epoch);
                }
            }
        }
    }
}

template <int Planes, bool Tight>
void launch(const uint32_t *src, uint32_t *dst, const Problem &p,
            TileQueue current, TileQueue next, int *stamp, int epoch,
            const int *dense, int tiles_x, int tile_count, int workers) {
    // Queue tiles remain coarse. Adjacent pairs use only 8 bytes of a 32-byte
    // sector per plane/row (up to 4x sector amplification before cache reuse).
    // This tradeoff needs GPU measurement; barrier removal does not prove a
    // throughput improvement over a wider, coalesced bit tile.
    if (p.boundary == Boundary::Torus) {
        if (p.rule.shape == Shape::Moore)
            step<Planes, true, true, Tight><<<workers, 128>>>(src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count);
        else step<Planes, false, true, Tight><<<workers, 128>>>(src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count);
    } else {
        if (p.rule.shape == Shape::Moore)
            step<Planes, true, false, Tight><<<workers, 128>>>(src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count);
        else step<Planes, false, false, Tight><<<workers, 128>>>(src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count);
    }
}

template <bool Tight>
void launch(const uint32_t *src, uint32_t *dst, const Problem &p,
            TileQueue current, TileQueue next, int *stamp, int epoch,
            const int *dense, int tiles_x, int tile_count, int workers) {
    switch (r1_multistate::plane_count(p.rule.states)) {
        case 1: launch<1, Tight>(src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count, workers); break;
        case 2: launch<2, Tight>(src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count, workers); break;
        case 3: launch<3, Tight>(src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count, workers); break;
        default: launch<4, Tight>(src, dst, p, current, next, stamp, epoch, dense, tiles_x, tile_count, workers); break;
    }
}
} // namespace r1_sparse_bits_gpu
#endif // __CUDACC__
