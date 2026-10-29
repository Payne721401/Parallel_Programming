#pragma once
#include "sparse_output.hpp"

// The caller retains the original pinned byte grid throughout evolution.
// Clear only tiles containing initial nonzero cells, then scatter all cells of
// each final queued tile. Unwritten cells were initially zero and are still
// zero by the sparse support invariant. Both reconstruction stages run on GPU.
namespace sparse_output_gpu {

struct Mapping {
    uint8_t *device = nullptr;
    const char *reason = "disabled";
};

struct Result {
    bool selected = false;
    const char *reason = "disabled";
    int dense = 0;
    std::size_t full_copy_bytes = 0;
    // Exact mapped bytes/counters are optional profile data. -1 means no
    // profiling counters were collected; it does not mean an empty grid.
    long long clear_tiles = -1, scatter_tiles = -1;
    unsigned long long clear_bytes = 0, scatter_bytes = 0;
};

inline cudaError_t prepare(uint8_t *host, bool requested, bool sparse_eligible, Mapping &mapping) {
    mapping = Mapping{};
    if (!requested) return cudaSuccess;
    mapping.reason = "not_sparse";
    if (!sparse_eligible) return cudaSuccess;
    cudaPointerAttributes attributes{};
    const cudaError_t error = cudaPointerGetAttributes(&attributes, host);
    if (error != cudaSuccess) {
        mapping.reason = "pointer_attributes";
        // Invalid host pointers are a supported fallback. Other errors may
        // describe a CUDA failure and must be handled by the caller.
        if (error != cudaErrorInvalidValue) return error;
        (void)cudaGetLastError();
        return cudaSuccess;
    }
    mapping.reason = attributes.type != cudaMemoryTypeHost ? "not_pinned_host" : "not_mapped";
    if (attributes.type != cudaMemoryTypeHost || !attributes.devicePointer) return cudaSuccess;
    mapping.device = static_cast<uint8_t *>(attributes.devicePointer);
    mapping.reason = "enabled";
    return cudaSuccess;
}

// Optional counters are allocated only for an explicitly profiled sample.
struct Counters {
    unsigned long long clear_tiles, scatter_tiles, clear_bytes, scatter_bytes;
};

__global__ void clear_initial(uint8_t *host, const uint8_t *initial_nonzero,
                              sparse_output::Geometry g, Counters *counters) {
    for (int tile = int(blockIdx.x); tile < g.tile_count; tile += int(gridDim.x)) {
        if (!initial_nonzero[tile]) continue; // Uniform over this block.
        for (int cell = int(threadIdx.x); cell < g.tile_width * g.tile_height; cell += int(blockDim.x)) {
            std::size_t index = 0;
            if (sparse_output::cell_index(g, tile, cell, index)) host[index] = 0;
        }
        if (counters && threadIdx.x == 0) {
            atomicAdd(&counters->clear_tiles, 1ull);
            atomicAdd(&counters->clear_bytes,
                      static_cast<unsigned long long>(sparse_output::tile_bytes(g, tile)));
        }
    }
}

__global__ void scatter_final(uint8_t *host, const uint8_t *src,
                              const int *ids, const int *count,
                              sparse_output::Geometry g, Counters *counters) {
    const int work_count = *count;
    for (int work = int(blockIdx.x); work < work_count; work += int(gridDim.x)) {
        const int tile = ids[work];
        for (int cell = int(threadIdx.x); cell < g.tile_width * g.tile_height; cell += int(blockDim.x)) {
            std::size_t index = 0;
            if (sparse_output::cell_index(g, tile, cell, index)) host[index] = src[index];
        }
        if (counters && threadIdx.x == 0) {
            atomicAdd(&counters->scatter_tiles, 1ull);
            atomicAdd(&counters->scatter_bytes,
                      static_cast<unsigned long long>(sparse_output::tile_bytes(g, tile)));
        }
    }
}

// Called after all generation work and any packed-to-byte unpack. Queue IDs
// must be unique and include every final nonzero cell's tile. initial_nonzero
// must remain unchanged since the initial input scan. T=0 returns earlier in
// run(); an empty final queue is supported here and clears vanished input.
//
// A dense fallback invalidates the final sparse queue, so read that flag once
// and use the ordinary full D2H path if it is set. Mapping failure also uses
// full D2H. This function completes all output writes before returning.
inline cudaError_t finish(uint8_t *host, const uint8_t *src, const Problem &p,
                           const uint8_t *initial_nonzero, const int *final_ids,
                           const int *final_count, const int *dense,
                           const sparse_output::Geometry &g, const Mapping &mapping,
                           Result &result, bool profile = false, cudaStream_t stream = nullptr) {
    result = Result{};
    result.reason = mapping.reason;
    const std::size_t bytes = std::size_t(p.width) * std::size_t(p.height);
    auto copy_all = [&]() {
        result.full_copy_bytes = bytes;
        cudaError_t error = cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost, stream);
        return error == cudaSuccess ? cudaStreamSynchronize(stream) : error;
    };
    if (!mapping.device) return copy_all();
    if (!sparse_output::valid(g) || g.width != p.width || g.height != p.height
        || !initial_nonzero || !final_ids || !final_count || !dense) {
        result.reason = "invalid_sparse_metadata";
        return copy_all();
    }
    cudaError_t error = cudaMemcpyAsync(&result.dense, dense, sizeof(result.dense),
                                      cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return error;
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return error;
    if (result.dense) {
        result.reason = "dense_fallback";
        return copy_all();
    }
    Counters *device_counters = nullptr;
    if (profile) {
        error = cudaMalloc(&device_counters, sizeof(Counters));
        if (error != cudaSuccess) return error;
        error = cudaMemsetAsync(device_counters, 0, sizeof(Counters), stream);
    }
    const int blocks = g.tile_count < 1024 ? g.tile_count : 1024;
    if (error == cudaSuccess) {
        clear_initial<<<blocks, 256, 0, stream>>>(mapping.device, initial_nonzero, g, device_counters);
        error = cudaGetLastError();
    }
    // Kernel ordering is required: overlapping clears and scatters can erase
    // valid final output even if each kernel has internally unique tiles.
    if (error == cudaSuccess) {
        scatter_final<<<blocks, 256, 0, stream>>>(mapping.device, src, final_ids, final_count, g, device_counters);
        error = cudaGetLastError();
    }
    Counters counters{};
    if (error == cudaSuccess && profile)
        error = cudaMemcpyAsync(&counters, device_counters, sizeof(Counters), cudaMemcpyDeviceToHost, stream);
    // Includes mapped host writes; returning before this wait would expose
    // partially reconstructed host memory to main()'s validation/hash scan.
    const cudaError_t sync_error = cudaStreamSynchronize(stream);
    if (error == cudaSuccess) error = sync_error;
    if (device_counters) {
        const cudaError_t free_error = cudaFree(device_counters);
        if (error == cudaSuccess) error = free_error;
    }
    if (error == cudaSuccess) {
        result.selected = true;
        result.reason = "enabled";
        if (profile) {
            result.clear_tiles = static_cast<long long>(counters.clear_tiles);
            result.scatter_tiles = static_cast<long long>(counters.scatter_tiles);
            result.clear_bytes = counters.clear_bytes;
            result.scatter_bytes = counters.scatter_bytes;
        }
    }
    return error;
}

} // namespace sparse_output_gpu
