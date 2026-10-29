#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include "../device_arena.hpp"

using device_arena::Layout;
using device_arena::Region;

static void require(bool ok, const char *what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL device arena: %s\n", what);
        std::exit(1);
    }
}

static std::vector<Region> regions(const Layout &layout) {
    return {layout.src, layout.dst, layout.nonzero, layout.ids[0], layout.ids[1],
            layout.counts, layout.stamp, layout.stats};
}

static void validate(const Layout &layout, std::size_t capacity,
                     bool sparse, std::size_t tiles, std::size_t stats) {
    require(layout.total_bytes && layout.total_bytes % layout.alignment == 0,
            "arena total alignment");
    require(layout.src.bytes == capacity && layout.dst.bytes == capacity,
            "both grids retain complete allocation capacity");
    require(layout.src.offset == 0, "source begins at arena base");
    std::size_t previous_end = 0;
    for (const Region &region : regions(layout)) {
        if (!region.present()) {
            require(region.bytes == 0, "absent regions have no payload");
            continue;
        }
        require(region.bytes > 0, "present regions have a payload");
        require(region.offset % layout.alignment == 0, "region alignment");
        require(region.offset >= previous_end, "regions do not overlap");
        require(region.offset <= layout.total_bytes &&
                region.bytes <= layout.total_bytes - region.offset,
                "region stays inside arena");
        previous_end = region.offset + region.bytes;
    }
    require(layout.nonzero.present() == sparse &&
            layout.ids[0].present() == sparse && layout.ids[1].present() == sparse &&
            layout.counts.present() == sparse && layout.stamp.present() == sparse,
            "dense/sparse metadata presence");
    require(layout.stats.present() == (stats != 0), "optional statistics presence");
    if (sparse) {
        require(layout.nonzero.bytes == tiles, "one initial flag per tile");
        require(layout.ids[0].bytes == tiles * sizeof(int) &&
                layout.ids[1].bytes == tiles * sizeof(int) &&
                layout.stamp.bytes == tiles * sizeof(int), "int metadata capacity");
        require(layout.counts.bytes == 4 * sizeof(int), "four queue/graph counters");
        require(layout.stats.bytes == stats, "caller-supplied statistics size");
    }
}

static void dense_and_padding() {
    Layout layout;
    require(device_arena::make_layout(1, false, 0, 0, layout), "one-byte dense grid");
    validate(layout, 1, false, 0, 0);
    require(layout.dst.offset == 256 && layout.total_bytes == 512,
            "one-byte grids remain separate aligned regions");

    // C16, W=3: four padded words per row need more than the byte grid. The
    // helper must preserve this supplied capacity rather than assume one byte.
    const std::size_t width = 3, height = 19, planes = 4;
    const std::size_t byte_payload = width * height;
    const std::size_t packed_payload = ((width + 31) / 32) * height * planes * 4;
    const std::size_t capacity = std::max(byte_payload, packed_payload);
    require(packed_payload > byte_payload, "narrow-grid padding test is active");
    require(device_arena::make_layout(capacity, false, 0, 0, layout), "padded dense grid");
    validate(layout, capacity, false, 0, 0);
    require(layout.src.bytes == 304 && layout.dst.offset == 512,
            "padded words fit without overlapping destination");

    // A wide packed representation is smaller, but byte input/output still use
    // the complete byte-grid capacity. No simulated allocation is needed here.
    const std::size_t bytes = std::size_t(32768) * 32768;
    require(device_arena::make_layout(bytes, false, 0, 0, layout), "large byte capacity");
    validate(layout, bytes, false, 0, 0);
    require(layout.total_bytes == bytes * 2, "large grid capacity remains uncompressed");
}

static void sparse_geometries() {
    const std::size_t width = 32768, height = 32768, capacity = width * height;
    const std::size_t narrow_tiles = ((width + 63) / 64) * ((height + 31) / 32);
    const std::size_t wide_tiles = ((width + 255) / 256) * ((height + 15) / 16);
    Layout narrow, wide, profiled;
    require(device_arena::make_layout(capacity, true, narrow_tiles, 0, narrow), "narrow metadata");
    require(device_arena::make_layout(capacity, true, wide_tiles, 0, wide), "wide metadata");
    require(device_arena::make_layout(capacity, true, narrow_tiles, 73, profiled), "profile metadata");
    validate(narrow, capacity, true, narrow_tiles, 0);
    validate(wide, capacity, true, wide_tiles, 0);
    validate(profiled, capacity, true, narrow_tiles, 73);
    require(wide_tiles * 2 == narrow_tiles && wide.total_bytes < narrow.total_bytes,
            "metadata reflects actual queue geometry");
    const auto plain_regions = regions(narrow), profiled_regions = regions(profiled);
    for (std::size_t i = 0; i + 1 < plain_regions.size(); ++i) {
        require(plain_regions[i].offset == profiled_regions[i].offset &&
                plain_regions[i].bytes == profiled_regions[i].bytes,
                "profile storage does not move existing buffers");
    }
}

static void region_writes() {
    std::mt19937 random(0xaffe1234u);
    for (int trial = 0; trial < 1000; ++trial) {
        const std::size_t capacity = 1 + random() % 2048;
        const bool sparse = trial % 3 != 0;
        const std::size_t tiles = sparse ? 1 + random() % 257 : 0;
        const std::size_t stats = sparse && trial % 2 ? 1 + random() % 161 : 0;
        const std::size_t alignment = trial % 5 ? 256 : 512;
        Layout layout;
        require(device_arena::make_layout(capacity, sparse, tiles, stats, layout, alignment),
                "random valid layout");
        validate(layout, capacity, sparse, tiles, stats);
        std::vector<unsigned char> storage(layout.total_bytes + 1, 0xa5);
        std::vector<unsigned char> expected = storage;
        unsigned char tag = 1;
        for (const Region &region : regions(layout)) {
            if (region.present()) {
                std::fill(storage.begin() + region.offset,
                          storage.begin() + region.offset + region.bytes, tag);
                for (std::size_t i = 0; i < region.bytes; ++i) {
                    require(expected[region.offset + i] == 0xa5, "write interval never overlaps");
                    expected[region.offset + i] = tag;
                }
            }
            ++tag;
        }
        require(storage == expected && storage.back() == 0xa5,
                "payload writes preserve all padding and final guard");
    }
}

static void invalid_requests_and_overflow() {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    std::size_t answer = 0;
    require(device_arena::checked_add(maximum - 1, 1, answer) && answer == maximum,
            "addition exact size_t limit");
    require(!device_arena::checked_add(maximum, 1, answer), "addition overflow");
    require(device_arena::checked_multiply(0, maximum, answer) && answer == 0,
            "zero multiplication");
    require(!device_arena::checked_multiply(maximum, 2, answer), "multiplication overflow");
    require(device_arena::checked_align_up(256, 256, answer) && answer == 256,
            "aligned value unchanged");
    require(device_arena::checked_align_up(257, 256, answer) && answer == 512,
            "unaligned value rounds upward");
    require(!device_arena::checked_align_up(maximum, 256, answer), "alignment overflow");
    for (std::size_t alignment : {std::size_t(0), std::size_t(1), std::size_t(128),
                                  std::size_t(255), std::size_t(384)}) {
        Layout layout;
        require(!device_arena::make_layout(1024, false, 0, 0, layout, alignment),
                "invalid arena alignment rejected");
    }
    struct Request { std::size_t capacity; bool sparse; std::size_t tiles, stats; };
    const Request invalid[] = {
        {0, false, 0, 0}, {1, true, 0, 0}, {1, false, 1, 0}, {1, false, 0, 1},
        {1, true, std::size_t(std::numeric_limits<int>::max()) + 1, 0},
        {maximum, false, 0, 0}, {maximum / 2 + 1, false, 0, 0},
        {1, true, 1, maximum},
    };
    for (const auto &request : invalid) {
        Layout layout;
        require(device_arena::make_layout(1024, true, 2, 8, layout), "seed successful output");
        require(!device_arena::make_layout(request.capacity, request.sparse,
                                           request.tiles, request.stats, layout),
                "invalid/overflowing request rejected");
        require(layout.total_bytes == 0, "failed layout clears total");
        for (const Region &region : regions(layout))
            require(!region.present() && region.bytes == 0, "failed layout exposes no partial regions");
    }
    Layout limit;
    require(!device_arena::make_layout(1, false, 0, 0, limit,
                                      std::size_t(1) << (sizeof(std::size_t) * 8 - 1)),
            "final arena rounding overflow rejected");
}

int main() {
    dense_and_padding();
    sparse_geometries();
    region_writes();
    invalid_requests_and_overflow();
    std::puts("device arena: capacity, geometry, alignment, writes, and overflow checks passed");
}
