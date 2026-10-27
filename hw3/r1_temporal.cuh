#pragma once
// Included inside hw3-1.cu's anonymous namespace after the packed R1 helpers.
// Dense only: each block independently computes a shrinking space-time region.
template <int K, bool Moore>
__global__ void step_r1_temporal(const uint8_t *__restrict__ input,
                                uint8_t *__restrict__ output, Problem p) {
    static_assert(K == 2 || K == 4 || K == 8, "bounded temporal specializations");
    // Round horizontal halo to whole four-byte words, but shrink only ONE cell
    // per side per generation. Partial words are masked after every update.
    constexpr int HX = ((K + 3) / 4) * 4;
    constexpr int RW = TILE_W + 2 * HX;
    constexpr int RH = TILE_H + 2 * K;
    constexpr int PITCH = RW / 4;
    constexpr int WORDS = PITCH * RH;
    static_assert(4 * WORDS * sizeof(uint32_t) <= 48 * 1024, "shared budget");
    __shared__ uint32_t storage[4 * WORDS];
    uint32_t *state = storage, *next = storage + WORDS;
    uint32_t *alive = storage + 2 * WORDS, *next_alive = storage + 3 * WORDS;
    const int tid = int(threadIdx.y) * 32 + int(threadIdx.x);
    const int x0 = int(blockIdx.x) * TILE_W, y0 = int(blockIdx.y) * TILE_H;
    const bool torus = p.boundary == Boundary::Torus;
    for (int i = tid; i < WORDS; i += 256) {
        const uint32_t s = load_r1_word(input, p, x0 - HX + (i % PITCH) * 4,
                                      y0 - K + i / PITCH);
        state[i] = s;
        alive[i] = r1_packed::equal(s, 1);
    }
    __syncthreads();
    for (int step = 1; step <= K; ++step) {
        const int first_word = step / 4;
        const int end_word = (RW - step + 3) / 4;
        const int previous_first = (step - 1) / 4;
        const int previous_end = (RW - (step - 1) + 3) / 4;
        const int columns = end_word - first_word;
        const int cells = columns * (RH - 2 * step);
        for (int i = tid; i < cells; i += 256) {
            const int wx = first_word + i % columns;
            const int sy = step + i / columns;
            const int q = sy * PITCH + wx;
            const bool left_valid = wx - 1 >= previous_first;
            const bool right_valid = wx + 1 < previous_end;
            uint32_t counts;
            if constexpr (Moore) {
                counts = r1_packed::count<true>(
                    left_valid ? alive[q-PITCH-1] : 0, alive[q-PITCH],
                    right_valid ? alive[q-PITCH+1] : 0,
                    left_valid ? alive[q-1] : 0, alive[q], right_valid ? alive[q+1] : 0,
                    left_valid ? alive[q+PITCH-1] : 0, alive[q+PITCH],
                    right_valid ? alive[q+PITCH+1] : 0, p.rule.count_centre);
            } else {
                counts = r1_packed::count<false>(0, alive[q-PITCH], 0,
                    left_valid ? alive[q-1] : 0, alive[q], right_valid ? alive[q+1] : 0,
                    0, alive[q+PITCH], 0, p.rule.count_centre);
            }
            const uint32_t s = r1_packed::next(state[q], counts, p.rule);
            uint32_t mask = 0;
            const int gy = y0 - K + sy;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int sx = wx * 4 + b, gx = x0 - HX + sx;
                // Dead-boundary halo must stay zero at EVERY intermediate step,
                // even if the rule could otherwise give birth outside the grid.
                if (sx >= step && sx < RW-step &&
                        (torus || (gx >= 0 && gx < p.width && gy >= 0 && gy < p.height)))
                    mask |= 0xffu << (8*b);
            }
            next[q] = s & mask;
            next_alive[q] = r1_packed::equal(s & mask, 1);
        }
        // All reads of state/alive and all writes of next/next_alive finish
        // before swapping their roles. No block can read another block's scratch.
        __syncthreads();
        uint32_t *old = state; state = next; next = old;
        old = alive; alive = next_alive; next_alive = old;
    }
    for (int i = tid; i < R1_WORDS * TILE_H; i += 256) {
        const int x = x0 + (i % R1_WORDS) * 4, y = y0 + i / R1_WORDS;
        if (x >= p.width || y >= p.height) continue;
        const uint32_t s = state[(K + i / R1_WORDS) * PITCH + HX/4 + i % R1_WORDS];
        const size_t index = size_t(y) * p.width + x;
        if ((p.width & 3) == 0) *reinterpret_cast<uint32_t *>(output + index) = s;
        else {
#pragma unroll
            for (int b = 0; b < 4; ++b)
                if (x+b < p.width) output[index+b] = uint8_t(s >> (8*b));
        }
    }
}
