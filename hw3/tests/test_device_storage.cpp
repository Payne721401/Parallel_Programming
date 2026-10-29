// Exercise the actual production ownership implementation with a CPU-backed
// CUDA allocation stub. Kernels, CUDA allocation latency, and device races
// remain the responsibility of the manual GPU/reference gate.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include "../device_arena.hpp"

static void require(bool ok, const char *what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL device storage: %s\n", what);
        std::exit(1);
    }
}

namespace mock_cuda {
alignas(256) static unsigned char heap[1 << 20];
static bool live = false;
static std::size_t allocations = 0, releases = 0, allocated_bytes = 0;
static void *released_pointer = nullptr;

static void reset() {
    require(!live, "prior ownership released before another scenario");
    allocations = releases = allocated_bytes = 0;
    released_pointer = nullptr;
}
} // namespace mock_cuda

template <class T>
static int cudaMalloc(T **out, std::size_t bytes) {
    require(!mock_cuda::live, "only one arena allocation per run");
    require(bytes > 0 && bytes <= sizeof(mock_cuda::heap), "mock allocation fits backing store");
    std::fill(std::begin(mock_cuda::heap), std::end(mock_cuda::heap), 0xa5);
    *out = reinterpret_cast<T *>(mock_cuda::heap);
    mock_cuda::live = true;
    mock_cuda::allocated_bytes = bytes;
    ++mock_cuda::allocations;
    return 0;
}

static int cudaFree(void *pointer) {
    require(mock_cuda::live, "allocation is released only once");
    require(pointer == mock_cuda::heap, "only the owning allocation base is freed");
    mock_cuda::live = false;
    mock_cuda::released_pointer = pointer;
    ++mock_cuda::releases;
    return 0;
}

static void check(int status, const char *) {
    require(status == 0, "mock CUDA call succeeded");
}

struct Problem { int width = 0, height = 0; };
struct SparseStats { std::uint64_t values[17]{}; };
struct SparseWorkspace {
    std::uint8_t *nonzero = nullptr;
    int *ids[2]{};
    int *counts = nullptr;
    int *stamp = nullptr;
    SparseStats *stats = nullptr;
    int tiles_x = 0, tile_count = 0;
};

// Include the allocator under test, rather than copying its implementation.
#include "../device_storage.cuh"

static_assert(!std::is_copy_constructible<RunStorage>::value, "owner cannot be copied");
static_assert(!std::is_copy_assignable<RunStorage>::value, "owner cannot be copy assigned");
static_assert(!std::is_move_constructible<RunStorage>::value, "owner has no unsafe implicit move");
static_assert(!std::is_move_assignable<RunStorage>::value, "owner has no unsafe implicit move assignment");

static void set_arena(const char *value) {
#ifdef _WIN32
    require(_putenv_s("PP_ARENA", value ? value : "") == 0, "set test arena environment");
#else
    require((value ? setenv("PP_ARENA", value, 1) : unsetenv("PP_ARENA")) == 0,
            "set test arena environment");
#endif
}

static void check_no_allocation_and_release() {
    for (const char *mode : {static_cast<const char *>(nullptr), "0", "1"}) {
        set_arena(mode);
        mock_cuda::reset();
        RunStorage unused;
        require(unused.requested == (mode && std::strcmp(mode, "1") == 0),
                "request follows environment");
        require(!unused.selected && !unused.owner && unused.layout.total_bytes == 0,
                "construction does not allocate device storage");
        unused.release();
        unused.release();
        unused.report(false);
        require(mock_cuda::allocations == 0 && mock_cuda::releases == 0 &&
                unused.allocation_calls == 0 && unused.release_calls == 0,
                "unused/early-return owner performs no CUDA allocation or release");
    }
    // The zero-step guard belongs to run(), which is not a CPU-only header.
    // This verifies the actual RunStorage constructed before that guard does
    // not allocate eagerly; it does not simulate or claim to test run() itself.
}

static void swap_grids_and_release(RunStorage &storage) {
    auto *src = storage.pointer<std::uint8_t>(storage.layout.src);
    auto *dst = storage.pointer<std::uint8_t>(storage.layout.dst);
    const auto *owner = storage.owner;
    require(src == mock_cuda::heap && dst != src, "grids bind to distinct arena regions");
    require(reinterpret_cast<std::uintptr_t>(src) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(dst) % 256 == 0, "grid pointers are aligned");
    for (int steps = 0; steps <= 1190; ++steps) {
        const auto &source_region = steps % 2 ? storage.layout.dst : storage.layout.src;
        require(src == mock_cuda::heap + source_region.offset,
                "source parity follows repeated grid swaps");
        require(storage.owner == owner, "grid swaps leave owning base unchanged");
        std::swap(src, dst);
    }
    storage.release();
    require(!storage.owner && mock_cuda::released_pointer == owner,
            "release uses original owner regardless of final source parity");
    storage.release();
    require(mock_cuda::allocations == 1 && mock_cuda::releases == 1 &&
            storage.allocation_calls == 1 && storage.release_calls == 1,
            "repeated release stays idempotent");
}

static void check_dense() {
    set_arena("1");
    for (bool profile : {false, true}) {
        mock_cuda::reset();
        RunStorage storage;
        // W=3,H=19,C16 requires 304 bytes after padded bit-plane rows.
        const std::size_t capacity = 304;
        storage.allocate(capacity, false, 0, profile);
        require(storage.requested && storage.selected && storage.owner,
                "requested arena becomes selected after allocation");
        require(mock_cuda::allocated_bytes == storage.layout.total_bytes,
                "one allocation uses exact computed arena total");
        require(storage.layout.src.bytes == capacity && storage.layout.dst.bytes == capacity,
                "both grids preserve narrow-row padded capacity");
        require(storage.pointer<int>(storage.layout.counts) == nullptr &&
                storage.pointer<SparseStats>(storage.layout.stats) == nullptr &&
                storage.pointer<std::uint8_t>(storage.layout.nonzero) == nullptr,
                "dense layout omits sparse metadata even with profiling");
        auto *src = storage.pointer<std::uint8_t>(storage.layout.src);
        auto *dst = storage.pointer<std::uint8_t>(storage.layout.dst);
        std::fill(src, src + capacity, 0x11);
        std::fill(dst, dst + capacity, 0x22);
        require(src[capacity - 1] == 0x11 && dst[0] == 0x22,
                "full padded grids can be written independently");
        swap_grids_and_release(storage);
    }
}

static void check_sparse() {
    set_arena("1");
    for (const Problem p : {Problem{1, 1}, Problem{65, 33}, Problem{257, 65}})
    for (bool wide : {false, true})
    for (bool profile : {false, true}) {
        mock_cuda::reset();
        RunStorage storage;
        const int tile_width = wide ? 256 : 64, tile_height = wide ? 16 : 32;
        const int tiles_x = (p.width + tile_width - 1) / tile_width;
        const int tile_count = tiles_x * ((p.height + tile_height - 1) / tile_height);
        const std::size_t capacity = std::max(std::size_t(p.width) * p.height,
                                             std::size_t((p.width + 31) / 32) * p.height * 16);
        storage.allocate(capacity, true, std::size_t(tile_count), profile);
        SparseWorkspace work;
        storage.bind_sparse(work, p, tile_width, tile_height);
        require(work.tiles_x == tiles_x && work.tile_count == tile_count,
                "binding retains actual narrow/wide partial-tile geometry");
        require(work.nonzero == storage.owner + storage.layout.nonzero.offset &&
                work.ids[0] == reinterpret_cast<int *>(storage.owner + storage.layout.ids[0].offset) &&
                work.ids[1] == reinterpret_cast<int *>(storage.owner + storage.layout.ids[1].offset) &&
                work.counts == reinterpret_cast<int *>(storage.owner + storage.layout.counts.offset) &&
                work.stamp == reinterpret_cast<int *>(storage.owner + storage.layout.stamp.offset),
                "workspace binds every metadata pointer to its actual layout region");
        require((work.stats != nullptr) == profile &&
                storage.layout.stats.bytes == (profile ? sizeof(SparseStats) : 0),
                "statistics storage follows profiling request");
        if (profile) {
            require(work.stats == reinterpret_cast<SparseStats *>(storage.owner + storage.layout.stats.offset),
                    "statistics bind to arena region");
            work.stats->values[16] = 0x123456789abcdef0ULL;
        }
        for (int tile = 0; tile < tile_count; ++tile) {
            work.nonzero[tile] = static_cast<std::uint8_t>(tile % 2);
            work.ids[0][tile] = tile;
            work.ids[1][tile] = tile_count - tile;
            work.stamp[tile] = 701 + tile;
        }
        for (int i = 0; i < 4; ++i) work.counts[i] = i + 11;
        for (int tile = 0; tile < tile_count; ++tile) {
            require(work.nonzero[tile] == tile % 2 && work.ids[0][tile] == tile &&
                    work.ids[1][tile] == tile_count - tile && work.stamp[tile] == 701 + tile,
                    "all metadata arrays remain independent under full-capacity writes");
        }
        require(work.counts[3] == 14 &&
                (!profile || work.stats->values[16] == 0x123456789abcdef0ULL),
                "counter/statistics writes survive neighboring metadata writes");
        swap_grids_and_release(storage);
    }
}

int main() {
    const char *previous = std::getenv("PP_ARENA");
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    check_no_allocation_and_release();
    check_dense();
    check_sparse();
    set_arena(had_previous ? saved.c_str() : nullptr);
    std::puts("device storage: production ownership, binding, parity, and release checks passed");
}
