// CUDA baseline: one thread per cell, one kernel per generation.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "ca.hpp"

namespace {

void check(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "hw3-1: %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Valid coordinates plus an offset in [-R, R] need at most one wrap because
// W,H >= 2R+1. This helper is NOT a general tile-halo coordinate map.
__device__ __forceinline__ int wrap_neighbour(int x, int extent) {
    if (x < 0) return x + extent;
    if (x >= extent) return x - extent;
    return x;
}

__device__ __forceinline__ uint8_t next_state(const Rule &r, uint8_t s, int n) {
    if (s == 0) return (r.birth_lo <= n && n <= r.birth_hi) ? 1 : 0;
    if (s == 1) return (r.survive_lo <= n && n <= r.survive_hi) ? 1 : (r.states == 2 ? 0 : 2);
    return s + 1 == r.states ? 0 : s + 1;
}

__global__ void step_naive(const uint8_t *__restrict__ src,
                           uint8_t *__restrict__ dst, Problem p) {
    const int x = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= p.width || y >= p.height) return;

    const Rule &r = p.rule;
    int n = 0;
    for (int dy = -r.radius; dy <= r.radius; ++dy) {
        int ny = y + dy;
        if (p.boundary == Boundary::Torus) ny = wrap_neighbour(ny, p.height);
        else if (ny < 0 || ny >= p.height) continue;

        const int abs_dy = dy < 0 ? -dy : dy;
        const int span = r.shape == Shape::Moore ? r.radius : r.radius - abs_dy;
        for (int dx = -span; dx <= span; ++dx) {
            if (dx == 0 && dy == 0 && !r.count_centre) continue;
            int nx = x + dx;
            if (p.boundary == Boundary::Torus) nx = wrap_neighbour(nx, p.width);
            else if (nx < 0 || nx >= p.width) continue;
            n += src[size_t(ny) * p.width + nx] == 1;
        }
    }
    const size_t index = size_t(y) * p.width + x;
    dst[index] = next_state(r, src[index], n);
}

} // namespace

void run(uint8_t *grid, const Problem &p) {
    if (p.steps == 0) return;
    const size_t bytes = size_t(p.width) * p.height;
    uint8_t *src = nullptr, *dst = nullptr;
    check(cudaMalloc(&src, bytes), "allocate source");
    check(cudaMalloc(&dst, bytes), "allocate destination");
    check(cudaMemcpy(src, grid, bytes, cudaMemcpyHostToDevice), "copy initial grid");

    const dim3 threads(32, 8);
    const dim3 blocks((p.width + 31) / 32, (p.height + 7) / 8);
    // One stream orders generations. Every output cell is written, so neither
    // a memset nor a host synchronization per generation is necessary.
    for (int t = 0; t < p.steps; ++t) {
        step_naive<<<blocks, threads>>>(src, dst, p);
        check(cudaGetLastError(), "launch step_naive");
        uint8_t *old_src = src;
        src = dst;
        dst = old_src;
    }
    // Blocking D2H waits for all preceding kernels; src holds the final state
    // for either parity of T. Asynchronous execution errors also surface here.
    check(cudaMemcpy(grid, src, bytes, cudaMemcpyDeviceToHost), "copy final grid");
    check(cudaFree(dst), "free destination");
    check(cudaFree(src), "free source");
}
