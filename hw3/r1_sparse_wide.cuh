#pragma once

// Include r1_sparse_wide.hpp outside the implementation namespace, and this
// header after SparseWorkspace, check(), and r1_multistate_gpu helpers.
// This owns a separate 256x16 queue. It never changes the established 64x32
// sparse path. Device counts/stamps are compatible with prepare_sparse().
namespace r1_sparse_wide_gpu {

__global__ void mark_initial(const uint8_t *src, uint8_t *nonzero, Problem p, int tiles_x) {
    const int tile = int(blockIdx.x);
    const int x0 = tile % tiles_x * r1_sparse_wide::width;
    const int y0 = tile / tiles_x * r1_sparse_wide::height;
    bool any = false;
    for (int cell = int(threadIdx.x); cell < r1_sparse_wide::width * r1_sparse_wide::height;
         cell += int(blockDim.x)) {
        const int x = x0 + cell % r1_sparse_wide::width;
        const int y = y0 + cell / r1_sparse_wide::width;
        if (x < p.width && y < p.height) any |= src[size_t(y) * p.width + x] != 0;
    }
    const bool found = __syncthreads_or(any);
    if (threadIdx.x == 0) nonzero[tile] = uint8_t(found);
}

__global__ void seed(const uint8_t *nonzero, TileQueue queue, Problem p,
                     int tiles_x, int tile_count) {
    const int tile = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tile >= tile_count) return;
    const bool torus = p.boundary == Boundary::Torus;
    const auto xs = tile_geometry::influence(tile % tiles_x, r1_sparse_wide::width, p.width, 1, torus);
    const auto ys = tile_geometry::influence(tile / tiles_x, r1_sparse_wide::height, p.height, 1, torus);
    bool needed = false;
    for (int y = 0; y < ys.count && !needed; ++y)
    for (int x = 0; x < xs.count; ++x)
        needed |= nonzero[ys.ids[y] * tiles_x + xs.ids[x]] != 0;
    if (needed) queue.ids[atomicAdd(queue.count, 1)] = tile;
}

struct Workspace : SparseWorkspace {
    void allocate(const Problem &p, bool profile) {
        tiles_x = (p.width + r1_sparse_wide::width - 1) / r1_sparse_wide::width;
        tile_count = tiles_x * ((p.height + r1_sparse_wide::height - 1) / r1_sparse_wide::height);
        check(cudaMalloc(&nonzero, size_t(tile_count)), "allocate wide initial flags");
        for (auto &list : ids) check(cudaMalloc(&list, size_t(tile_count) * sizeof(int)), "allocate wide list");
        check(cudaMalloc(&counts, 4 * sizeof(int)), "allocate wide counters");
        check(cudaMalloc(&stamp, size_t(tile_count) * sizeof(int)), "allocate wide stamps");
        if (profile) check(cudaMalloc(&stats, sizeof(SparseStats)), "allocate wide statistics");
    }

    bool initialize(const uint8_t *src, uint8_t *dst, const Problem &p, bool forced) {
        check(cudaMemsetAsync(counts, 0, 4 * sizeof(int)), "initialize wide counters");
        mark_initial<<<tile_count, 256>>>(src, nonzero, p, tiles_x);
        check(cudaGetLastError(), "launch mark_initial wide");
        seed<<<(tile_count + 255) / 256, 256>>>(nonzero, queue(0), p, tiles_x, tile_count);
        check(cudaGetLastError(), "launch seed wide");
        check(cudaMemcpy(&initial_work, counts, sizeof(int), cudaMemcpyDeviceToHost), "read wide initial count");
        if (!forced && initial_work > tile_count / 4) return false;
        check(cudaMemsetAsync(dst, 0, size_t(p.width) * p.height), "clear wide sparse destination");
        check(cudaMemsetAsync(stamp, 0, size_t(tile_count) * sizeof(int)), "clear wide stamps");
        if (stats) check(cudaMemsetAsync(stats, 0, sizeof(SparseStats)), "clear wide statistics");
        int device = 0, sms = 0;
        check(cudaGetDevice(&device), "get wide CUDA device");
        check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "get wide SM count");
        workers = tile_count < 4 * sms ? tile_count : 4 * sms;
        return true;
    }
};

// All 32 lanes participate, including lanes without a candidate. Winners
// reserve one contiguous range per warp, rather than serial count atomics.
__device__ __forceinline__ void enqueue(const tile_geometry::AxisTiles &xs,
                                       const tile_geometry::AxisTiles &ys, int self,
                                       TileQueue next, int *stamp, int epoch, int tiles_x) {
    const int lane = int(threadIdx.x) & 31;
    const int tile = r1_sparse_wide::candidate(lane, xs, ys, tiles_x, self);
    const bool winner = tile >= 0 && atomicExch(stamp + tile, epoch) != epoch;
    const unsigned mask = __ballot_sync(0xffffffffu, winner);
    int start = 0;
    if (lane == 0 && mask) start = atomicAdd(next.count, __popc(mask));
    start = __shfl_sync(0xffffffffu, start, 0);
    if (winner) {
        const unsigned preceding = lane == 0 ? 0 : mask & ((1u << lane) - 1u);
        next.ids[start + __popc(preceding)] = tile;
    }
}

template <int Planes, bool Torus>
__device__ __forceinline__ r1_multistate_gpu::Row<Planes>
load_row(const uint32_t *src, int wx, int y, const Problem &p) {
    const int column = r1_sparse_wide::column(int(threadIdx.x) & 31);
    const int pitch = (p.width + 31) / 32;
    const auto state = r1_multistate_gpu::load_word<Planes, Torus>(src, wx, y, pitch, p.height);
    const uint32_t a = r1_multistate::alive(state, p.rule.states);
    uint32_t previous = __shfl_up_sync(0xffffffffu, a, 1, r1_sparse_wide::words_per_row);
    uint32_t following = __shfl_down_sync(0xffffffffu, a, 1, r1_sparse_wide::words_per_row);
    if (column == 0) {
        const int x = Torus && wx == 0 ? pitch - 1 : wx - 1;
        previous = r1_multistate::alive(
            r1_multistate_gpu::load_word<Planes, Torus>(src, x, y, pitch, p.height), p.rule.states);
    }
    if (column == r1_sparse_wide::words_per_row - 1 || wx == pitch - 1) {
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
    const int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) / 32;
    const bool full = *dense != 0;
    const int count = full ? tile_count : *current.count;
    const int pitch = (p.width + 31) / 32;
    const size_t words = size_t(pitch) * p.height;
    for (int work = int(blockIdx.x) * r1_sparse_wide::warps_per_block + warp;
         work < count; work += int(gridDim.x) * r1_sparse_wide::warps_per_block) {
        const int tile = full ? work : current.ids[work];
        const int x0 = tile % tiles_x * r1_sparse_wide::width;
        const int y0 = tile / tiles_x * r1_sparse_wide::height;
        const int wx = x0 / 32 + r1_sparse_wide::column(lane);
        const int y_first = y0 + r1_sparse_wide::first_row(lane);
        auto upper = load_row<Planes, Torus>(src, wx, y_first - 1, p);
        auto middle = load_row<Planes, Torus>(src, wx, y_first, p);
        r1_sparse_wide::Bounds bounds;
        unsigned flags = 0;
#pragma unroll
        for (int row = 0; row < r1_sparse_wide::rows_per_lane; ++row) {
            const int y = y_first + row;
            const auto lower = load_row<Planes, Torus>(src, wx, y + 1, p);
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
                const uint32_t tail = r1_bits::mask(p.width, wx);
                const size_t index = size_t(y) * pitch + wx;
                uint32_t before = 0, after = 0;
#pragma unroll
                for (int b = 0; b < Planes; ++b) {
                    dst[size_t(b) * words + index] = output.bit[b] & tail;
                    before |= middle.state.bit[b];
                    after |= output.bit[b];
                }
                if (!full) {
                    flags |= unsigned((before & tail) != 0) | (unsigned((after & tail) != 0) << 1);
                    if constexpr (Tight) {
                        const uint32_t live = r1_multistate::alive(output, p.rule.states) & tail;
                        if (live) {
                            bounds.include(wx * 32 - x0 + __ffs(live) - 1, y - y0);
                            bounds.include(wx * 32 - x0 + 31 - __clz(live), y - y0);
                        }
                    }
                }
            }
            upper = middle;
            middle = lower;
        }
        if (!full) {
            flags = __reduce_or_sync(0xffffffffu, flags);
            tile_geometry::AxisTiles xs, ys;
            if constexpr (Tight) {
                bounds.xmin = __reduce_min_sync(0xffffffffu, bounds.xmin);
                bounds.ymin = __reduce_min_sync(0xffffffffu, bounds.ymin);
                bounds.xmax = __reduce_max_sync(0xffffffffu, bounds.xmax);
                bounds.ymax = __reduce_max_sync(0xffffffffu, bounds.ymax);
                if (!bounds.empty()) {
                    xs = tile_geometry::influence_interval(x0 + bounds.xmin, x0 + bounds.xmax + 1,
                        r1_sparse_wide::width, p.width, 1, Torus);
                    ys = tile_geometry::influence_interval(y0 + bounds.ymin, y0 + bounds.ymax + 1,
                        r1_sparse_wide::height, p.height, 1, Torus);
                }
            } else if (flags & 2) {
                xs = tile_geometry::influence(tile % tiles_x, r1_sparse_wide::width, p.width, 1, Torus);
                ys = tile_geometry::influence(tile / tiles_x, r1_sparse_wide::height, p.height, 1, Torus);
            }
            const int self = (flags & (Tight ? 3u : 1u)) ? tile : -1;
            enqueue(xs, ys, self, next, stamp, epoch, tiles_x);
        }
    }
}

template <int Planes, bool Tight>
void launch(const uint32_t *src, uint32_t *dst, const Problem &p,
            TileQueue current, TileQueue next, int *stamp, int epoch,
            const int *dense, int tiles_x, int tile_count, int workers) {
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
} // namespace r1_sparse_wide_gpu
