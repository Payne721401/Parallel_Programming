// Included inside the implementation namespace, after reduce_alive_bounds.
// Only thread 0 consumes the result. The caller's tile-end barrier protects reuse.
template <bool Tight, int Arch>
__device__ __forceinline__ unsigned finish_tracking(unsigned flags, AliveBounds b, AliveBounds *support) {
    if constexpr ((Arch & 1) == 0) {
        const unsigned input = __syncthreads_or((flags & 1) != 0);
        const unsigned output = __syncthreads_or((flags & 2) != 0);
        if constexpr (Tight) *support = reduce_alive_bounds(b);
        return input | (output << 1);
    } else {
        // Plain fields: __shared__ objects must not require a constructor.
        struct Summary { unsigned flags; int xmin, ymin, xmax, ymax; };
        __shared__ Summary summaries[8];
        const int lane = int(threadIdx.x), warp = int(threadIdx.y);
        // H100 supports native warp integer reductions. All 32 lanes participate.
        flags = __reduce_or_sync(0xffffffffu, flags);
        if constexpr (Tight) {
            b.xmin = __reduce_min_sync(0xffffffffu, b.xmin);
            b.ymin = __reduce_min_sync(0xffffffffu, b.ymin);
            b.xmax = __reduce_max_sync(0xffffffffu, b.xmax);
            b.ymax = __reduce_max_sync(0xffffffffu, b.ymax);
        }
        if (lane == 0) {
            summaries[warp].flags = flags;
            if constexpr (Tight) {
                summaries[warp].xmin = b.xmin; summaries[warp].ymin = b.ymin;
                summaries[warp].xmax = b.xmax; summaries[warp].ymax = b.ymax;
            }
        }
        __syncthreads(); // publish both flags and bounds together
        if (lane == 0 && warp == 0) {
            flags = 0;
            if constexpr (Tight) b = AliveBounds{};
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                flags |= summaries[w].flags;
                if constexpr (Tight) {
                    const Summary other = summaries[w];
                    if (other.xmax >= other.xmin) {
                        b.include(other.xmin, other.ymin);
                        b.include(other.xmax, other.ymax);
                    }
                }
            }
            if constexpr (Tight) *support = b;
        }
        return flags;
    }
}
