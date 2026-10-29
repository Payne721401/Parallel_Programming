// Include inside the implementation namespace, after arch_stats.cuh. Requires
// rolling_filters.hpp, r1_packed.hpp, next_state, wrap_halo, finish_tracking.
// Exactly 256 threads, dim3(32,8), participate in every block-wide barrier.

__device__ __forceinline__ uint8_t rolling_source_cell(
    const uint8_t *src, const Problem &p, int x, int y) {
    if (p.boundary == Boundary::Torus) {
        x = wrap_halo(x, p.width); y = wrap_halo(y, p.height);
    }
    if (x < 0 || x >= p.width || y < 0 || y >= p.height) return 0;
    return src[std::size_t(y) * p.width + x];
}

__device__ __forceinline__ uint32_t rolling_source_word(
    const uint8_t *src, const Problem &p, int x, int y) {
    if (p.boundary == Boundary::Torus) y = wrap_halo(y, p.height);
    if (y < 0 || y >= p.height) return 0;
    // Never issue a misaligned vector access; odd widths and torus seams pack
    // four scalar bytes, including partly out-of-grid dead-boundary words.
    int gx = x;
    if (p.boundary == Boundary::Torus) gx = wrap_halo(gx, p.width);
    if (gx >= 0 && gx + 3 < p.width) {
        const std::size_t index = std::size_t(y) * p.width + gx;
        if ((index & 3) == 0)
            return *reinterpret_cast<const uint32_t *>(src + index);
    }
    uint32_t word = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b)
        word |= uint32_t(rolling_source_cell(src, p, x + b, y)) << (8 * b);
    return word;
}

// All inputs are at most 65, so their signed-byte interpretation is unchanged.
__device__ __forceinline__ unsigned rolling_sum_bytes(uint32_t word) {
    return unsigned(__dp4a(int(word), int(0x01010101), 0));
}
__device__ __forceinline__ unsigned rolling_partial_bytes(uint32_t word, int count) {
    const uint32_t mask = count ? (0xffffffffu >> (32 - 8 * count)) : 0u;
    return rolling_sum_bytes(word & mask);
}
__device__ __forceinline__ unsigned rolling_prefix_endpoint(
    const uint32_t *vertical, const uint16_t *prefix, int endpoint) {
    const int word = endpoint / 4, bytes = endpoint & 3;
    return unsigned(prefix[word]) + (bytes ? rolling_partial_bytes(vertical[word], bytes) : 0u);
}

template <bool Moore, bool Track, bool Tight = false, int Arch = 0,
          int Width = 64, int Height = 32>
__device__ unsigned update_rolling(const uint8_t *__restrict__ src,
    uint8_t *__restrict__ dst, const Problem &p, int x0, int y0,
    uint8_t *scratch, AliveBounds *support = nullptr) {
    static_assert(Width % 4 == 0 && Height % 8 == 0, "whole packed words and warp rows");
    static_assert(!Tight || Track, "support requires tracking");
    const int lane = int(threadIdx.x), warp = int(threadIdx.y), tid = 32 * warp + lane;
    const int r = p.rule.radius, left = rolling_filters::halo_x(r);
    const int pitch = rolling_filters::source_pitch<Width, Height>(r);
    const int word_pitch = pitch / 4, source_height = Height + 2 * r;
    auto *source_words = reinterpret_cast<uint32_t *>(scratch);
    for (int word = tid; word < word_pitch * source_height; word += 256) {
        const int row = word / word_pitch, x = (word % word_pitch) * 4;
        source_words[word] = rolling_source_word(src, p, x0 + x - left, y0 + row - r);
    }
    __syncthreads();
    const uint8_t *source = scratch;
    uint8_t *intermediate = scratch + rolling_filters::source_bytes<Width, Height>(r);
    unsigned flags = 0;
    AliveBounds bounds;

    if constexpr (Moore) {
        // One register holds four independent vertical counts. Each owner
        // streams up to 16 rows; the complete source tile is loaded only once.
        auto *vertical_words = reinterpret_cast<uint32_t *>(intermediate);
        constexpr int segments = (Height + rolling_filters::line_rows - 1) / rolling_filters::line_rows;
        for (int task = tid; task < word_pitch * segments; task += 256) {
            const int column = task % word_pitch;
            const int first = (task / word_pitch) * rolling_filters::line_rows;
            const int end = first + rolling_filters::line_rows < Height
                ? first + rolling_filters::line_rows : Height;
            uint32_t sum = 0;
            for (int dy = 0; dy <= 2 * r; ++dy)
                sum += r1_packed::equal(source_words[(first + dy) * word_pitch + column], 1);
            for (int y = first; y < end; ++y) {
                vertical_words[y * word_pitch + column] = sum;
                if (y + 1 < end)
                    sum = rolling_filters::slide_bytes(sum,
                        r1_packed::equal(source_words[y * word_pitch + column], 1),
                        r1_packed::equal(source_words[(y + 2 * r + 1) * word_pitch + column], 1));
            }
        }
        __syncthreads();
        // Prefix only the sums of four columns, not each padded source cell.
        // A warp owns each complete row, including its private prefix scratch.
        auto *prefix_base = reinterpret_cast<uint16_t *>(intermediate + pitch * Height);
        uint16_t *prefix = prefix_base + warp * (word_pitch + 1);
        for (int y = warp; y < Height; y += 8) {
            const uint32_t *v = vertical_words + y * word_pitch;
            if (lane == 0) prefix[0] = 0;
            unsigned carry = 0;
            for (int chunk = 0; chunk < word_pitch; chunk += 32) {
                const int column = chunk + lane;
                unsigned value = column < word_pitch ? rolling_sum_bytes(v[column]) : 0u;
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2) {
                    const unsigned other = __shfl_up_sync(0xffffffffu, value, offset);
                    if (lane >= offset) value += other;
                }
                if (column < word_pitch) prefix[column + 1] = uint16_t(carry + value);
                carry += __shfl_sync(0xffffffffu, value, 31);
            }
            __syncwarp();
            for (int word = lane; word < Width / 4; word += 32) {
                const int first_x = word * 4;
                const uint32_t state_word = source_words[(y + r) * word_pitch + first_x / 4 + left / 4];
                uint32_t output_word = 0;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int x = first_x + b, gx = x0 + x, gy = y0 + y;
                    const uint8_t state = uint8_t(state_word >> (8 * b));
                    const int centre = left + x;
                    int n = int(rolling_prefix_endpoint(v, prefix, centre + r + 1))
                          - int(rolling_prefix_endpoint(v, prefix, centre - r));
                    if (!p.rule.count_centre && state == 1) --n;
                    const uint8_t output = next_state(p.rule, state, n);
                    output_word |= uint32_t(output) << (8 * b);
                    if (gx < p.width && gy < p.height) {
                        if constexpr (Tight) if (output == 1) bounds.include(x, y);
                        if constexpr (Track) flags |= unsigned(state != 0) | (unsigned(output != 0) << 1);
                    }
                }
                const int gx = x0 + first_x, gy = y0 + y;
                if (gx < p.width && gy < p.height) {
                    const std::size_t index = std::size_t(gy) * p.width + gx;
                    if ((index & 3) == 0 && gx + 3 < p.width)
                        *reinterpret_cast<uint32_t *>(dst + index) = output_word;
                    else {
#pragma unroll
                        for (int b = 0; b < 4; ++b)
                            if (gx + b < p.width) dst[index + b] = uint8_t(output_word >> (8 * b));
                    }
                }
            }
            __syncwarp(); // no owner may reuse prefix storage before all readers
        }
    } else {
        // Two exact diagonal boxes cover the diamond's two parity sublattices.
        // U is computed in an extended domain: dead-boundary U outside the grid
        // is NOT automatically zero because its source window can enter it.
        const int iw = Width + r, ih = Height + r;
        const int upitch = rolling_filters::align2(iw);
        auto *u = reinterpret_cast<uint16_t *>(intermediate);
        const int diagonals = iw + ih - 1;
        for (int diagonal = tid; diagonal < diagonals; diagonal += 256) {
            int x = diagonal < iw ? diagonal : 0;
            int y = diagonal < iw ? 0 : diagonal - iw + 1;
            int sum = 0;
            for (int i = 0; i <= r; ++i)
                sum += source[(y + i) * pitch + left - r + x + i] == 1;
            while (x < iw && y < ih) {
                const unsigned last = source[(y + r) * pitch + left + x] == 1;
                u[y * upitch + x] = rolling_filters::pack_diagonal(unsigned(sum), last);
                if (x + 1 >= iw || y + 1 >= ih) break;
                sum -= source[y * pitch + left - r + x] == 1;
                sum += source[(y + r + 1) * pitch + left + x + 1] == 1;
                ++x; ++y;
            }
        }
        __syncthreads();
        // Bounded anti-diagonal strips retain coalesced output for their
        // interior lanes. Each output after initialization needs four U reads.
        constexpr int rows = rolling_filters::line_rows;
        constexpr int starts = (Width + rows - 1 + 31) & ~31;
        constexpr int strips = (Height + rows - 1) / rows;
        for (int task = tid; task < starts * strips; task += 256) {
            const int strip = task / starts;
            const int start_x = task % starts - (rows - 1);
            const int top = strip * rows;
            const int bottom = top + rows < Height ? top + rows - 1 : Height - 1;
            const int count = bottom - top + 1;
            const int first = start_x < 0 ? -start_x : 0;
            const int end = Width - start_x < count ? Width - start_x : count;
            if (first >= end) continue;
            int x = start_x + first, y = bottom - first;
            int a = 0, b = 0;
            for (int j = 0; j <= r; ++j)
                a += rolling_filters::diagonal_plus(u[(y + r - j) * upitch + x + j]);
            for (int j = 0; j < r; ++j)
                b += rolling_filters::diagonal_minus(u[(y + r - j) * upitch + x + 1 + j]);
            for (int step = first; step < end; ++step) {
                const int gx = x0 + x, gy = y0 + y;
                const uint8_t state = source[(y + r) * pitch + left + x];
                const int n = a + b - int(!p.rule.count_centre && state == 1);
                const uint8_t output = next_state(p.rule, state, n);
                if (gx < p.width && gy < p.height) {
                    dst[std::size_t(gy) * p.width + gx] = output;
                    if constexpr (Tight) if (output == 1) bounds.include(x, y);
                    if constexpr (Track) flags |= unsigned(state != 0) | (unsigned(output != 0) << 1);
                }
                if (step + 1 < end) {
                    a -= rolling_filters::diagonal_plus(u[(y + r) * upitch + x]);
                    a += rolling_filters::diagonal_plus(u[(y - 1) * upitch + x + r + 1]);
                    b -= rolling_filters::diagonal_minus(u[(y + r) * upitch + x + 1]);
                    b += rolling_filters::diagonal_minus(u[y * upitch + x + r + 1]);
                    ++x; --y;
                }
            }
        }
    }
    if constexpr (Track) return finish_tracking<Tight, Arch>(flags, bounds, support);
    return 0;
}

template <bool Moore>
__global__ void step_rolling(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, Problem p) {
    extern __shared__ uint32_t rolling_scratch[];
    update_rolling<Moore, false, false, 0,
        rolling_filters::dense_width, rolling_filters::dense_height>(src, dst, p,
        int(blockIdx.x) * rolling_filters::dense_width,
        int(blockIdx.y) * rolling_filters::dense_height,
        reinterpret_cast<uint8_t *>(rolling_scratch));
}
