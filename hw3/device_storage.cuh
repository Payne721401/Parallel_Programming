#pragma once

// Host-only ownership. Include after SparseWorkspace/check and include
// device_arena.hpp outside the implementation namespace. No device kernels.
struct RunStorage {
    bool requested = false, selected = false;
    uint8_t *owner = nullptr;
    device_arena::Layout layout{};
    size_t allocation_calls = 0, release_calls = 0;

    RunStorage() {
        const char *value = std::getenv("PP_ARENA");
        requested = value && std::strcmp(value, "1") == 0;
        if (value && !requested && std::strcmp(value, "0") != 0) {
            std::fprintf(stderr, "hw3-1: PP_ARENA must be 0 or 1\n");
            std::exit(EXIT_FAILURE);
        }
    }
    RunStorage(const RunStorage &) = delete;
    RunStorage &operator=(const RunStorage &) = delete;

    void allocate(size_t grid_capacity, bool inspect_sparse, size_t tile_count, bool profile) {
        if (!device_arena::make_layout(grid_capacity, inspect_sparse, tile_count,
                inspect_sparse && profile ? sizeof(SparseStats) : 0, layout)) {
            std::fprintf(stderr, "hw3-1: invalid or overflowing device arena layout\n");
            std::exit(EXIT_FAILURE);
        }
        check(cudaMalloc(&owner, layout.total_bytes), "allocate run storage arena");
        selected = true;
        allocation_calls = 1;
    }

    template <class T> T *pointer(const device_arena::Region &region) const {
        return region.present() ? reinterpret_cast<T *>(owner + region.offset) : nullptr;
    }

    void bind_sparse(SparseWorkspace &work, const Problem &p, int tile_width, int tile_height) const {
        work.tiles_x = (p.width + tile_width - 1) / tile_width;
        work.tile_count = work.tiles_x * ((p.height + tile_height - 1) / tile_height);
        if (layout.nonzero.bytes != size_t(work.tile_count)) {
            std::fprintf(stderr, "hw3-1: device arena workspace geometry mismatch\n");
            std::exit(EXIT_FAILURE);
        }
        work.nonzero = pointer<uint8_t>(layout.nonzero);
        work.ids[0] = pointer<int>(layout.ids[0]);
        work.ids[1] = pointer<int>(layout.ids[1]);
        work.counts = pointer<int>(layout.counts);
        work.stamp = pointer<int>(layout.stamp);
        work.stats = pointer<SparseStats>(layout.stats);
    }

    void release() {
        if (owner) {
            check(cudaFree(owner), "free run storage arena");
            owner = nullptr;
            ++release_calls;
        }
    }

    void report(bool profile) const {
        if (profile) std::fprintf(stderr,
            "[PP_MEMORY] arena_requested=%d arena_selected=%d arena_bytes=%zu "
            "storage_allocations=%zu storage_frees=%zu\n",
            int(requested), int(selected), selected ? layout.total_bytes : size_t(0),
            allocation_calls, release_calls);
    }
};
