// CPU reconstruction proof. Shares production tile indexing; actual mapped
// host stores, CUDA pointer selection and GPU ordering require GPU gates.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include "../sparse_output.hpp"

static unsigned long long comparisons = 0;
static unsigned cases = 0;
static void require(bool value, const char *message) {
    if (!value) { std::fprintf(stderr, "FAIL sparse output CPU test: %s\n", message); std::exit(1); }
}

static int tile_of(const sparse_output::Geometry &g, int x, int y) {
    return (y / g.tile_height) * g.tiles_x + x / g.tile_width;
}

static void reconstruct(int width, int height, int tile_width, int tile_height,
                        int states, int pattern, std::mt19937 &rng) {
    const auto g = sparse_output::geometry(width, height, tile_width, tile_height);
    require(sparse_output::valid(g), "valid test geometry");
    const std::size_t cells = std::size_t(width) * height;
    std::vector<uint8_t> initial(cells), final(cells), flags(std::size_t(g.tile_count));
    std::vector<uint8_t> final_flags(std::size_t(g.tile_count));
    for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
        const std::size_t index = std::size_t(y) * width + x;
        if (pattern == 0 || pattern == 2 || pattern == 3)
            initial[index] = rng() % 97 == 0 ? uint8_t(1 + rng() % (states - 1)) : 0;
        if (pattern == 1 || pattern == 2 || pattern == 3)
            final[index] = rng() % 131 == 0 ? uint8_t(1 + rng() % (states - 1)) : 0;
        if (pattern == 4) { // Edge/tail births, extinction and arbitrary decay.
            initial[index] = x == 0 || y == 0 ? uint8_t(states - 1) : 0;
            final[index] = x == width - 1 || y == height - 1 ? uint8_t(states - 1) : 0;
        }
        flags[std::size_t(tile_of(g, x, y))] |= initial[index] != 0;
        final_flags[std::size_t(tile_of(g, x, y))] |= final[index] != 0;
    }
    std::vector<int> queue;
    for (int tile = 0; tile < g.tile_count; ++tile)
        if (final_flags[std::size_t(tile)] || (pattern == 3 && rng() % 3 == 0)) queue.push_back(tile);
    std::shuffle(queue.begin(), queue.end(), rng);
    auto host = initial;
    std::size_t total_tile_bytes = 0;
    std::vector<uint8_t> visits(cells);
    for (int tile = 0; tile < g.tile_count; ++tile) {
        total_tile_bytes += sparse_output::tile_bytes(g, tile);
        for (int cell = 0; cell < tile_width * tile_height; ++cell) {
            std::size_t index = 0;
            if (!sparse_output::cell_index(g, tile, cell, index)) continue;
            require(index < cells && ++visits[index] == 1, "unique in-bounds tile ownership");
            if (flags[std::size_t(tile)]) host[index] = 0;
        }
    }
    require(total_tile_bytes == cells, "clipped tile bytes cover exact grid");
    for (int tile : queue)
    for (int cell = 0; cell < tile_width * tile_height; ++cell) {
        std::size_t index = 0;
        if (sparse_output::cell_index(g, tile, cell, index)) host[index] = final[index];
    }
    for (std::size_t i = 0; i < cells; ++i) {
        require(visits[i] == 1 && host[i] == final[i], "final reconstruction including stale initial values");
        ++comparisons;
    }
    ++cases;
}

int main() {
    require(!sparse_output::valid(sparse_output::geometry(0, 3, 64, 32)), "zero width rejected");
    require(!sparse_output::valid(sparse_output::geometry(3, 0, 64, 32)), "zero height rejected");
    require(!sparse_output::valid(sparse_output::geometry(3, 3, 0, 32)), "zero tile rejected");
    std::mt19937 rng(0x53434154u);
    const std::pair<int, int> grids[] = {{3,3},{31,7},{64,32},{65,33},{129,65},
        {255,15},{256,16},{257,17},{511,63},{513,65},{1025,129}};
    for (const auto &tile : {std::pair<int,int>{64,32}, {256,16}})
    for (const auto &grid : grids)
    for (int states : {2,3,4,5,8,16})
    for (int pattern = 0; pattern < 6; ++pattern) // pattern5 is all-zero input/output.
        reconstruct(grid.first, grid.second, tile.first, tile.second, states, pattern, rng);
    for (const auto &tile : {std::pair<int,int>{64,32}, {256,16}}) {
        const auto g = sparse_output::geometry(32768, 32768, tile.first, tile.second);
        std::size_t index = 0;
        require(sparse_output::cell_index(g, g.tile_count - 1,
            tile.first * tile.second - 1, index), "largest-grid final cell exists");
        require(index == std::size_t(32768) * 32768 - 1, "largest-grid index exact");
        require(!sparse_output::cell_index(g, -1, 0, index), "negative tile rejected");
        require(!sparse_output::cell_index(g, g.tile_count, 0, index), "tile outside grid rejected");
    }
    std::printf("PASS sparse output: %u cases, %llu cell comparisons; CUDA mapping/stores unverified\n",
                cases, comparisons);
}
