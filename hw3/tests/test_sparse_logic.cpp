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

static void model(Problem p, std::vector<uint8_t> src, bool tight) {
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
            tile_geometry::Bounds bounds;
            const int x0 = (tile % nx)*64, y0 = (tile / nx)*32;
            for (int y = y0; y < std::min(y0+32, p.height); ++y)
                for (int x = x0; x < std::min(x0+64, p.width); ++x) {
                    const size_t i = size_t(y)*p.width+x;
                    in |= src[i] != 0;
                    out |= expected[i] != 0;
                    if (expected[i] == 1) bounds.include(x-x0, y-y0);
                    dst[i] = expected[i];
                }
            if (tight) {
                if (!bounds.empty()) {
                    const auto xs = tile_geometry::influence_interval(x0+bounds.xmin, x0+bounds.xmax+1,
                        64, p.width, p.rule.radius, p.boundary == Boundary::Torus);
                    const auto ys = tile_geometry::influence_interval(y0+bounds.ymin, y0+bounds.ymax+1,
                        32, p.height, p.rule.radius, p.boundary == Boundary::Torus);
                    for (int y = 0; y < ys.count; ++y)
                        for (int x = 0; x < xs.count; ++x) next.insert(ys.ids[y]*nx+xs.ids[x]);
                }
                if (out) next.insert(tile); // decay without any alive neighbours
            } else if (out) {
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
            // Independent interval oracle: singleton endpoints, interior and
            // full intervals, including every partial final tile and R=1..32.
            const int begin = tile*size, end = std::min((tile+1)*size, extent);
            const int intervals[][2] = {{begin,begin+1},{end-1,end},
                {begin+(end-begin)/3,begin+2*(end-begin)/3+1}};
            for (const auto &interval : intervals) {
                const auto narrowed = tile_geometry::influence_interval(interval[0],interval[1],size,extent,radius,torus);
                std::set<int> narrowed_set(narrowed.ids,narrowed.ids+narrowed.count), oracle;
                for (int pos=interval[0]-radius; pos<interval[1]+radius; ++pos) {
                    const int x = torus ? (pos%extent+extent)%extent : pos;
                    if (x>=0 && x<extent) oracle.insert(x/size);
                }
                require(narrowed.count <= 4 && int(narrowed_set.size()) == narrowed.count
                    && narrowed_set == oracle, "bbox interval influence set");
                require(std::includes(got.begin(),got.end(),narrowed_set.begin(),narrowed_set.end()),
                    "bbox must be a subset of whole-tile influence");
                ++axis_checks;
            }
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
        for (bool tight : {false, true}) { model(p, src, tight); ++models; }
    }
    // Same population, different layout; mixed alive/decay and cleanup over
    // repeated buffer reuse. No CUDA fallback in this model, so skips are tested.
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
    for (int pattern = 0; pattern < 3; ++pattern) {
        Problem p{193,97,Rule{1,4,false,2,3,3,3,Shape::Moore},boundary,19};
        std::vector<uint8_t> src(size_t(p.width)*p.height,0);
        for (int k=0; k<9; ++k) {
            const int x=pattern==0 ? 20+k%3 : pattern==1 ? (k%3)*64 : (191+k%3)%p.width;
            const int y=pattern==0 ? 12+k/3 : pattern==1 ? (k/3)*32 : (95+k/3)%p.height;
            src[size_t(y)*p.width+x]=1;
        }
        src[31*p.width+63]=2;
        for (bool tight : {false,true}) { model(p,src,tight); ++models; }
    }
    std::printf("PASS %d axis influence checks; %d sparse buffer models (CPU only)\n", axis_checks, models);
}
