#pragma once

// Include r1_multistate.hpp outside the implementation namespace first. These
// kernels may then be included inside hw3-1.cu's anonymous namespace. They own
// no allocation/events and leave the existing dense C2 kernel unchanged.
namespace r1_multistate_gpu {

template <int Planes>
__global__ void pack_kernel(const uint8_t *src, uint32_t *dst, Problem p) {
    const int lane = int(threadIdx.x) & 31;
    const int pitch = (p.width + 31) / 32;
    const std::size_t words = std::size_t(pitch) * p.height;
    for (std::size_t word = std::size_t(blockIdx.x) * (int(blockDim.x) / 32)
                            + int(threadIdx.x) / 32;
         word < words; word += std::size_t(gridDim.x) * (int(blockDim.x) / 32)) {
        const int y = int(word / pitch), x = int(word % pitch) * 32 + lane;
        const unsigned state = x < p.width ? src[std::size_t(y) * p.width + x] : 0;
#pragma unroll
        for (int b = 0; b < Planes; ++b) {
            const uint32_t bits = __ballot_sync(0xffffffffu, (state >> b) & 1u);
            if (lane == 0) dst[std::size_t(b) * words + word] = bits;
        }
    }
}

template <int Planes>
__global__ void unpack_kernel(const uint32_t *src, uint8_t *dst, Problem p) {
    const int lane = int(threadIdx.x) & 31;
    const int pitch = (p.width + 31) / 32;
    const std::size_t words = std::size_t(pitch) * p.height;
    for (std::size_t word = std::size_t(blockIdx.x) * (int(blockDim.x) / 32)
                            + int(threadIdx.x) / 32;
         word < words; word += std::size_t(gridDim.x) * (int(blockDim.x) / 32)) {
        r1_multistate::Word<Planes> state;
#pragma unroll
        for (int b = 0; b < Planes; ++b) state.bit[b] = src[std::size_t(b) * words + word];
        const int y = int(word / pitch), x = int(word % pitch) * 32 + lane;
        if (x < p.width)
            dst[std::size_t(y) * p.width + x] = uint8_t(r1_multistate::extract(state, lane));
    }
}

template <int Planes, bool Torus>
__device__ __forceinline__ r1_multistate::Word<Planes>
load_word(const uint32_t *src, int wx, int y, int pitch, int height) {
    r1_multistate::Word<Planes> result;
    if (wx < 0 || wx >= pitch) return result;
    if constexpr (Torus) {
        if (y < 0) y += height;
        else if (y >= height) y %= height;
    } else if (y < 0 || y >= height) return result;
    const std::size_t words = std::size_t(pitch) * height;
    const std::size_t index = std::size_t(y) * pitch + wx;
#pragma unroll
    for (int b = 0; b < Planes; ++b) result.bit[b] = src[std::size_t(b) * words + index];
    return result;
}

template <int Planes> struct Row {
    r1_multistate::Word<Planes> state;
    uint32_t left, centre, right;
};

template <int Planes, bool Torus>
__device__ __forceinline__ Row<Planes>
load_row(const uint32_t *src, int wx, int y, const Problem &p) {
    const int lane = int(threadIdx.x) & 31, pitch = (p.width + 31) / 32;
    const auto state = load_word<Planes, Torus>(src, wx, y, pitch, p.height);
    const uint32_t a = r1_multistate::alive(state, p.rule.states);
    uint32_t prev = __shfl_up_sync(0xffffffffu, a, 1);
    uint32_t next = __shfl_down_sync(0xffffffffu, a, 1);
    if (lane == 0)
        prev = r1_multistate::alive(load_word<Planes, Torus>(src, wx - 1, y, pitch, p.height), p.rule.states);
    if (lane == 31)
        next = r1_multistate::alive(load_word<Planes, Torus>(src, wx + 1, y, pitch, p.height), p.rule.states);
    if constexpr (Torus) {
        if (wx == 0)
            prev = r1_multistate::alive(load_word<Planes, true>(src, pitch - 1, y, pitch, p.height), p.rule.states);
        if (wx == pitch - 1)
            next = r1_multistate::alive(load_word<Planes, true>(src, 0, y, pitch, p.height), p.rule.states);
    }
    const auto shifted = r1_multistate::horizontal(a, prev, next, wx, p.width, Torus);
    return {state, shifted.left, shifted.centre, shifted.right};
}

template <int Planes, int Rows, bool Moore, bool Torus>
__global__ void step_kernel(const uint32_t *src, uint32_t *dst, Problem p) {
    const int wx = int(blockIdx.x) * 32 + int(threadIdx.x);
    const int y0 = (int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y)) * Rows;
    const int pitch = (p.width + 31) / 32;
    const std::size_t words = std::size_t(pitch) * p.height;
    // Single-generation spatial reuse; all lanes enter every shuffle, even
    // padded words/rows. Only valid output words are stored and tail-masked.
    auto up = load_row<Planes, Torus>(src, wx, y0 - 1, p);
    auto mid = load_row<Planes, Torus>(src, wx, y0, p);
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        const int y = y0 + row;
        auto down = load_row<Planes, Torus>(src, wx, y + 1, p);
        r1_bits::Count count;
        if constexpr (Moore)
            count = r1_bits::count9(up.left, up.centre, up.right, mid.left, mid.right,
                                   down.left, down.centre, down.right,
                                   p.rule.count_centre ? mid.centre : 0);
        else
            count = r1_bits::count9(0, up.centre, 0, mid.left, mid.right, 0,
                                   down.centre, 0, p.rule.count_centre ? mid.centre : 0);
        const auto out = r1_multistate::next(mid.state, count, p.rule);
        if (wx < pitch && y < p.height) {
            const uint32_t mask = r1_bits::mask(p.width, wx);
            const std::size_t index = std::size_t(y) * pitch + wx;
#pragma unroll
            for (int b = 0; b < Planes; ++b)
                dst[std::size_t(b) * words + index] = out.bit[b] & mask;
        }
        up = mid;
        mid = down;
    }
}

inline int conversion_blocks(const Problem &p) {
    const std::size_t requested = (r1_multistate::plane_words(p) + 7) / 8;
    return int(requested < 4096 ? requested : 4096);
}

template <int Planes>
inline void launch_step(const uint32_t *src, uint32_t *dst, const Problem &p,
                        cudaStream_t stream) {
    constexpr int Rows = 4;
    const dim3 threads(32, 4);
    const dim3 blocks(((p.width + 31) / 32 + 31) / 32, (p.height + 4 * Rows - 1) / (4 * Rows));
    if (p.rule.shape == Shape::Moore) {
        if (p.boundary == Boundary::Torus)
            step_kernel<Planes, Rows, true, true><<<blocks, threads, 0, stream>>>(src, dst, p);
        else step_kernel<Planes, Rows, true, false><<<blocks, threads, 0, stream>>>(src, dst, p);
    } else {
        if (p.boundary == Boundary::Torus)
            step_kernel<Planes, Rows, false, true><<<blocks, threads, 0, stream>>>(src, dst, p);
        else step_kernel<Planes, Rows, false, false><<<blocks, threads, 0, stream>>>(src, dst, p);
    }
}

// Source and destination must be different allocations. Their required size is
// r1_multistate::allocation_bytes(p), which also accommodates the byte grid.
inline cudaError_t pack(const uint8_t *src, uint32_t *dst, const Problem &p,
                        cudaStream_t stream = nullptr) {
    const int blocks = conversion_blocks(p);
    switch (r1_multistate::plane_count(p.rule.states)) {
        case 1: pack_kernel<1><<<blocks, 256, 0, stream>>>(src, dst, p); break;
        case 2: pack_kernel<2><<<blocks, 256, 0, stream>>>(src, dst, p); break;
        case 3: pack_kernel<3><<<blocks, 256, 0, stream>>>(src, dst, p); break;
        default: pack_kernel<4><<<blocks, 256, 0, stream>>>(src, dst, p); break;
    }
    return cudaGetLastError();
}

inline cudaError_t step(const uint32_t *src, uint32_t *dst, const Problem &p,
                        cudaStream_t stream = nullptr) {
    switch (r1_multistate::plane_count(p.rule.states)) {
        case 1: launch_step<1>(src, dst, p, stream); break;
        case 2: launch_step<2>(src, dst, p, stream); break;
        case 3: launch_step<3>(src, dst, p, stream); break;
        default: launch_step<4>(src, dst, p, stream); break;
    }
    return cudaGetLastError();
}

inline cudaError_t unpack(const uint32_t *src, uint8_t *dst, const Problem &p,
                          cudaStream_t stream = nullptr) {
    const int blocks = conversion_blocks(p);
    switch (r1_multistate::plane_count(p.rule.states)) {
        case 1: unpack_kernel<1><<<blocks, 256, 0, stream>>>(src, dst, p); break;
        case 2: unpack_kernel<2><<<blocks, 256, 0, stream>>>(src, dst, p); break;
        case 3: unpack_kernel<3><<<blocks, 256, 0, stream>>>(src, dst, p); break;
        default: unpack_kernel<4><<<blocks, 256, 0, stream>>>(src, dst, p); break;
    }
    return cudaGetLastError();
}

} // namespace r1_multistate_gpu
