// Four byte cells per lane. Each half warp owns 64 columns and two output rows.
// No shared-memory tile. The upper/current/lower rows slide through registers.
struct R1WarpRow { uint32_t state, left, centre, right; };
__device__ R1WarpRow load_r1_warp_row(const uint8_t *src, const Problem &p, int x, int y) {
    const int half_lane = int(threadIdx.x) & 15;
    const uint32_t state = load_r1_word(src, p, x, y);
    const uint32_t alive = r1_packed::equal(state, 1);
    uint32_t left = __shfl_up_sync(0xffffffffu, alive, 1, 16);
    uint32_t right = __shfl_down_sync(0xffffffffu, alive, 1, 16);
    if (half_lane == 0) left = r1_packed::equal(load_r1_word(src, p, x-4, y), 1);
    if (half_lane == 15) right = r1_packed::equal(load_r1_word(src, p, x+4, y), 1);
    return {state, left, alive, right};
}

template <bool Moore, bool Track, bool Tight, int Arch>
__device__ unsigned update_r1_warp(const uint8_t *src, uint8_t *dst, const Problem &p,
                                  int x0, int y0, AliveBounds *support) {
    const int lane = int(threadIdx.x), warp = int(threadIdx.y);
    const int x = x0 + (lane & 15)*4;
    const int first_y = y0 + (warp*2 + lane/16)*2;
    auto upper = load_r1_warp_row(src, p, x, first_y-1);
    auto middle = load_r1_warp_row(src, p, x, first_y);
    auto lower = load_r1_warp_row(src, p, x, first_y+1);
    AliveBounds bounds;
    unsigned flags = 0;
#pragma unroll
    for (int row = 0; row < 2; ++row) {
        const int y = first_y + row;
        const uint32_t counts = r1_packed::count<Moore>(
            upper.left, upper.centre, upper.right,
            middle.left, middle.centre, middle.right,
            lower.left, lower.centre, lower.right, p.rule.count_centre);
        const uint32_t output = r1_packed::next(middle.state, counts, p.rule);
        if (x < p.width && y < p.height) {
            const int valid = p.width-x < 4 ? p.width-x : 4;
            const uint32_t mask = valid == 4 ? 0xffffffffu : (1u << (8*valid))-1u;
            const size_t index = size_t(y)*p.width+x;
            if ((p.width & 3) == 0) *reinterpret_cast<uint32_t *>(dst+index) = output;
            else {
#pragma unroll
                for (int b = 0; b < 4; ++b)
                    if (b < valid) dst[index+b] = uint8_t(output >> (8*b));
            }
            if constexpr (Tight) {
#pragma unroll
                for (int b = 0; b < 4; ++b)
                    if (b < valid && uint8_t(output >> (8*b)) == 1)
                        bounds.include(x-x0+b, y-y0);
            }
            if constexpr (Track)
                flags |= unsigned((middle.state & mask) != 0) | (unsigned((output & mask) != 0) << 1);
        }
        // Invalid output lanes also execute every shuffle with the full mask.
        if (row == 0) {
            upper = middle; middle = lower;
            lower = load_r1_warp_row(src, p, x, first_y+2);
        }
    }
    if constexpr (Track) return finish_tracking<Tight, Arch>(flags, bounds, support);
    return 0;
}
