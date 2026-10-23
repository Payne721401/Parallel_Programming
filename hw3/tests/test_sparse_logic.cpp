// CPU-only proof checks of tile geometry and sparse buffer/list invariants.
// This does NOT compile or execute CUDA; test_hw3-1 --sparse checks real kernels.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>
#include "../tile_geometry.hpp"
#define run run_reference
#include "../original_version/hw3-1.cu"
#undef run

static void require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

static std::set<int> neighbours(int tile, const Problem &p) {
    const int nx = (p.width + 63) / 64;
    const bool torus = p.boundary == Boundary::Torus;
    const auto xs = tile_geometry::influence(tile % nx, 64, p.width, p.rule.radius, torus);
    const auto ys = tile_geometry::influence(tile / nx, 32, p.height, p.rule.radius, torus);
    std::set<int> result;
    for (int y = 0; y < ys.count; ++y)
        for (int x = 0; x < xs.count; ++x) result.insert(ys.ids[y] * nx + xs.ids[x]);
    return result;
}

static void model(Problem p, std::vector<uint8_t> src) {
    const int nx = (p.width + 63) / 64, ny = (p.height + 31) / 32;
    std::vector<uint8_t> dst(src.size(), 0);
    std::set<int> initial, work;
    for (int y = 0; y < p.height; ++y)
        for (int x = 0; x < p.width; ++x)
            if (src[size_t(y)*p.width+x]) initial.insert((y/32)*nx+x/64);
    for (int tile : initial) {
        const auto ns = neighbours(tile, p);
        work.insert(ns.begin(), ns.end());
    }
    // Seed kernel uses inverse influence. Check it agrees with forward marking.
    std::set<int> inverse;
    for (int tile = 0; tile < nx*ny; ++tile)
        for (int other : neighbours(tile, p))
            if (initial.count(other)) { inverse.insert(tile); break; }
    require(work == inverse, "forward/inverse initial dilation differs");
    const int steps = p.steps;
    p.steps = 1;
    for (int t = 0; t < steps; ++t) {
        auto expected = src;
        run_reference(expected.data(), p);
        std::set<int> next;
        for (int tile : work) {
            bool in = false, out = false;
            const int x0 = (tile % nx)*64, y0 = (tile / nx)*32;
            for (int y = y0; y < std::min(y0+32, p.height); ++y)
                for (int x = x0; x < std::min(x0+64, p.width); ++x) {
                    const size_t i = size_t(y)*p.width+x;
                    in |= src[i] != 0;
                    out |= expected[i] != 0;
                    dst[i] = expected[i];
                }
            if (out) {
                const auto ns = neighbours(tile, p);
                next.insert(ns.begin(), ns.end());
            }
            if (in) next.insert(tile); // old src is next generation's destination
        }
        require(dst == expected, "sparse worklist missed a change or stale value");
        src.swap(dst);
        work.swap(next);
    }
}

int main() {
    int axis_checks = 0, models = 0;
    for (int size : {32, 64})
    for (int radius = 1; radius <= 32; ++radius)
    for (int extent : {2*radius+1, 2*radius+3, 65, 66, 95, 97, 127, 129, 193, 257, 321, 32768}) {
        if (extent < 2*radius+1) continue;
        for (bool torus : {false, true})
        for (int tile = 0; tile < (extent+size-1)/size; ++tile) {
            const auto actual = tile_geometry::influence(tile, size, extent, radius, torus);
            require(actual.count > 0 && actual.count <= 4, "axis capacity");
            std::set<int> got(actual.ids, actual.ids + actual.count), expected;
            // Independent cell-by-cell oracle, including partial-tile wrapping.
            for (int pos = tile*size-radius; pos < std::min((tile+1)*size, extent)+radius; ++pos) {
                int x = pos;
                if (torus) x = (x % extent + extent) % extent;
                if (x >= 0 && x < extent) expected.insert(x/size);
            }
            require(int(got.size()) == actual.count && got == expected, "axis influence set");
            ++axis_checks;
        }
    }
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
    for (int radius : {1, 5})
    for (Shape shape : {Shape::Moore, Shape::VonNeumann})
    for (int pattern = 0; pattern < 3; ++pattern) {
        const int maxn = shape == Shape::Moore ? (2*radius+1)*(2*radius+1)-1 : 2*radius*(radius+1);
        Rule r{radius, pattern == 2 ? 16 : 2, false, 2, std::min(3,maxn), 1, 1, shape};
        if (pattern == 0) { r.survive_lo = maxn; r.survive_hi = maxn; r.birth_lo = maxn; r.birth_hi = maxn; }
        Problem p{193, 97, r, boundary, pattern == 2 ? 17 : 5};
        std::vector<uint8_t> src(size_t(p.width)*p.height, 0);
        for (int y : {0, 31, 32, 96})
            for (int x : {0, 63, 64, 192}) src[size_t(y)*p.width+x] = pattern == 2 ? 2 : 1;
        model(p, src);
        ++models;
    }
    std::printf("PASS %d axis influence checks; %d sparse buffer models (CPU only)\n", axis_checks, models);
}
