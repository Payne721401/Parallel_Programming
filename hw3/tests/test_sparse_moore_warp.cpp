// CPU validation of the production register-window arithmetic and strip
// ownership. This does not validate CUDA shuffles, races or GPU performance.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "../ca.hpp"
#include "../r1_packed.hpp"
#include "../sparse_moore_warp.hpp"
#include "../tile_geometry.hpp"

using Grid = std::vector<uint8_t>;
static unsigned long long count_checks = 0, direct_checks = 0, state_checks = 0;
static unsigned sparse_steps = 0, dense_steps = 0, cleanup_steps = 0;

static void require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
static uint8_t cell(const Grid &grid, const Problem &p, int x, int y) {
    if (p.boundary == Boundary::Torus) {
        x = (x % p.width + p.width) % p.width;
        y = (y % p.height + p.height) % p.height;
    }
    return x < 0 || x >= p.width || y < 0 || y >= p.height ? 0 : grid[y * p.width + x];
}
static uint32_t word(const Grid &grid, const Problem &p, int x, int y) {
    uint32_t value = 0;
    for (int b = 0; b < 4; ++b) value |= uint32_t(cell(grid, p, x + b, y)) << (8 * b);
    return value;
}
static uint8_t next(uint8_t old, int count, const Rule &rule) {
    if (old >= 2) return old == rule.states - 1 ? 0 : old + 1;
    if (old == 1)
        return count >= rule.survive_lo && count <= rule.survive_hi ? 1 : rule.states == 2 ? 0 : 2;
    return count >= rule.birth_lo && count <= rule.birth_hi ? 1 : 0;
}
static int direct(const Grid &grid, const Problem &p, int x, int y) {
    int sum = 0;
    for (int dy = -p.rule.radius; dy <= p.rule.radius; ++dy)
        for (int dx = -p.rule.radius; dx <= p.rule.radius; ++dx)
            if (dx || dy || p.rule.count_centre) sum += cell(grid, p, x + dx, y + dy) == 1;
    return sum;
}

// Independent scalar2D integral oracle. Its samples are also checked by direct
// neighbourhood enumeration, including repeated wraps on partial tile halos.
static std::vector<int> oracle(const Grid &grid, const Problem &p, std::mt19937 &rng) {
    const int r = p.rule.radius, width = p.width + 2 * r, height = p.height + 2 * r;
    std::vector<int> prefix((width + 1) * (height + 1)), result(grid.size());
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            prefix[(y + 1) * (width + 1) + x + 1] = int(cell(grid, p, x - r, y - r) == 1)
                + prefix[(y + 1) * (width + 1) + x] + prefix[y * (width + 1) + x + 1]
                - prefix[y * (width + 1) + x];
    for (int y = 0; y < p.height; ++y)
        for (int x = 0; x < p.width; ++x) {
            const int hi_x = x + 2 * r + 1, hi_y = y + 2 * r + 1;
            result[y * p.width + x] = prefix[hi_y * (width + 1) + hi_x]
                - prefix[hi_y * (width + 1) + x] - prefix[y * (width + 1) + hi_x]
                + prefix[y * (width + 1) + x] - int(!p.rule.count_centre && grid[y * p.width + x] == 1);
        }
    for (int sample = 0; sample < 24; ++sample) {
        const int x = sample < 4 ? ((sample & 1) ? p.width - 1 : 0) : int(rng() % p.width);
        const int y = sample < 4 ? ((sample & 2) ? p.height - 1 : 0) : int(rng() % p.height);
        require(result[y * p.width + x] == direct(grid, p, x, y), "integral vs direct oracle");
        ++direct_checks;
    }
    return result;
}

struct StripSummary { unsigned flags = 0; tile_geometry::Bounds bounds; };
static StripSummary emulate_strip(const Grid &src, Grid &dst, const Problem &p, int tile, int strip,
                                  const std::vector<int> &expected) {
    const int tiles_x = (p.width + 63) / 64, x0 = tile % tiles_x * 64, y0 = tile / tiles_x * 32;
    const int r = p.rule.radius, left = sparse_moore_warp::halo_x(r);
    const int wp = sparse_moore_warp::word_pitch(r), first_y = strip * sparse_moore_warp::rows_per_warp;
    std::array<uint32_t, 32> vertical{};
    StripSummary summary;
    if (y0 + first_y >= p.height) return summary;
    require(wp <= 32, "source fits one warp");
    for (int lane = 0; lane < wp; ++lane)
        for (int dy = -r; dy <= r; ++dy)
            vertical[lane] += r1_packed::equal(word(src, p, x0 + lane * 4 - left, y0 + first_y + dy), 1);
    for (int row = 0; row < sparse_moore_warp::rows_per_warp; ++row) {
        const int y = first_y + row, gy = y0 + y;
        std::array<unsigned, 33> prefix{};
        for (int lane = 0; lane < 32; ++lane)
            for (int b = 0; b < 4; ++b) prefix[lane + 1] += (vertical[lane] >> (8 * b)) & 255u;
        for (int lane = 0; lane < 32; ++lane) prefix[lane + 1] += prefix[lane];
        for (int lane = 0; lane < 16; ++lane) {
            const int lo = left + lane * 4 - r, hi = left + lane * 4 + r + 1;
            const int lw = lo / 4, hw = hi / 4;
            require(lw >= 0 && lw < 32 && hw >= 0 && hw < 32, "shuffle endpoint index");
            const auto counts = sparse_moore_warp::window4(prefix[lw], vertical[lw],
                lw + 1 < wp ? vertical[lw + 1] : 0, lo & 3,
                prefix[hw], vertical[hw], hw + 1 < wp ? vertical[hw + 1] : 0, hi & 3);
            for (int b = 0; b < 4; ++b) {
                const int x = lane * 4 + b, gx = x0 + x;
                if (gx >= p.width || gy >= p.height) continue;
                const int index = gy * p.width + gx;
                const uint8_t old = src[index];
                const int n = int(counts.cell[b]) - int(!p.rule.count_centre && old == 1);
                if (n != expected[index]) {
                    std::fprintf(stderr, "R%d C%d boundary%d xy%d,%d expected%d got%d\n", r,
                        p.rule.states, int(p.boundary), gx, gy, expected[index], n);
                    std::exit(1);
                }
                ++count_checks;
                const uint8_t out = next(old, n, p.rule);
                dst[index] = out;
                summary.flags |= unsigned(old != 0) | (unsigned(out != 0) << 1);
                if (out == 1) summary.bounds.include(x, y);
            }
        }
        if (row + 1 < sparse_moore_warp::rows_per_warp)
            for (int lane = 0; lane < wp; ++lane)
                vertical[lane] = sparse_moore_warp::slide(vertical[lane],
                    r1_packed::equal(word(src, p, x0 + lane * 4 - left, gy - r), 1),
                    r1_packed::equal(word(src, p, x0 + lane * 4 - left, gy + r + 1), 1));
    }
    return summary;
}

static void append(std::vector<bool> &queue, const tile_geometry::AxisTiles &xs,
                   const tile_geometry::AxisTiles &ys, int tiles_x) {
    for (int y = 0; y < ys.count; ++y)
        for (int x = 0; x < xs.count; ++x) queue[ys.ids[y] * tiles_x + xs.ids[x]] = true;
}
static void enqueue(std::vector<bool> &queue, const StripSummary &s, const Problem &p,
                    int tile, bool tight) {
    const int tx = (p.width + 63) / 64, x0 = tile % tx * 64, y0 = tile / tx * 32;
    const bool torus = p.boundary == Boundary::Torus;
    if (tight) {
        if (!s.bounds.empty())
            append(queue, tile_geometry::influence_interval(x0 + s.bounds.xmin, x0 + s.bounds.xmax + 1,
                       64, p.width, p.rule.radius, torus),
                   tile_geometry::influence_interval(y0 + s.bounds.ymin, y0 + s.bounds.ymax + 1,
                       32, p.height, p.rule.radius, torus), tx);
        if (s.flags & 3) queue[tile] = true;
    } else {
        if (s.flags & 2)
            append(queue, tile_geometry::influence(tile % tx, 64, p.width, p.rule.radius, torus),
                   tile_geometry::influence(tile / tx, 32, p.height, p.rule.radius, torus), tx);
        if (s.flags & 1) queue[tile] = true;
    }
}

static void arithmetic(std::mt19937 &rng) {
    for (int r = 2; r <= 32; ++r)
        for (int boundary = 0; boundary < 2; ++boundary)
            for (int centre = 0; centre < 2; ++centre)
                for (int pattern = 0; pattern < 3; ++pattern) {
                    Problem p{};
                    p.width = 2 * r + 1 + (r % 7); p.height = 2 * r + 1 + (r % 5);
                    if (pattern == 2) { p.width += 64; p.height += 32; }
                    p.rule = {r, 2 + r % 15, bool(centre), centre, (2 * r + 1) * (2 * r + 1) - 1 + centre,
                              1, 1 + r, Shape::Moore};
                    p.boundary = boundary ? Boundary::Torus : Boundary::Dead;
                    require(sparse_moore_warp::eligible(p), "eligibility");
                    Grid src(p.width * p.height), dst(src.size());
                    for (auto &v : src) v = pattern == 0 ? 0 : pattern == 1 ? 1 : uint8_t(rng() % p.rule.states);
                    const auto expected = oracle(src, p, rng);
                    const int tiles = ((p.width + 63) / 64) * ((p.height + 31) / 32);
                    for (int tile = 0; tile < tiles; ++tile)
                        for (int strip = 0; strip < sparse_moore_warp::strips_per_tile; ++strip)
                            emulate_strip(src, dst, p, tile, strip, expected);
                    for (std::size_t i = 0; i < src.size(); ++i) {
                        require(dst[i] == next(src[i], expected[i], p.rule), "state update");
                        ++state_checks;
                    }
                }
}

static void evolution(std::mt19937 &rng) {
    for (int boundary = 0; boundary < 2; ++boundary)
        for (int tight = 0; tight < 2; ++tight)
            for (int pattern = 0; pattern < 4; ++pattern)
                for (int states : {2, 3, 4, 16}) {
                    Problem p{};
                    p.width = 321; p.height = 129; p.boundary = boundary ? Boundary::Torus : Boundary::Dead;
                    const int r = pattern == 3 ? 32 : pattern == 2 ? 2 : 5;
                    const int maxn = (2 * r + 1) * (2 * r + 1) - 1;
                    p.rule = {r, states, bool(pattern & 1), pattern & 1, maxn + (pattern & 1),
                              1, pattern == 2 ? maxn : 3, Shape::Moore};
                    const int tx = (p.width + 63) / 64, tiles = tx * ((p.height + 31) / 32);
                    Grid reference(p.width * p.height), buffers[2] = {Grid(reference.size()), Grid(reference.size())};
                    if (pattern == 0) {
                        for (int y = 60; y < 65; ++y) for (int x = 156; x < 164; ++x)
                            reference[y * p.width + x] = uint8_t(rng() % states);
                    } else if (pattern == 1) {
                        reference[0] = uint8_t(states - 1);
                        reference[31 * p.width + 63] = uint8_t(states - 1);
                        reference.back() = uint8_t(states - 1);
                    } else if (pattern == 2) {
                        reference[64 * p.width + 160] = 1;
                    } else {
                        for (auto &v : reference) v = uint8_t(rng() % states);
                    }
                    buffers[0] = reference;
                    std::vector<bool> work(tiles);
                    for (int y = 0; y < p.height; ++y) for (int x = 0; x < p.width; ++x)
                        if (reference[y * p.width + x]) {
                            const int tile = y / 32 * tx + x / 64;
                            append(work, tile_geometry::influence(tile % tx, 64, p.width, r, boundary != 0),
                                         tile_geometry::influence(tile / tx, 32, p.height, r, boundary != 0), tx);
                        }
                    bool full = false;
                    for (int gen = 0; gen < 8; ++gen) {
                        const auto expected = oracle(reference, p, rng);
                        Grid expected_grid(reference.size());
                        for (std::size_t i = 0; i < reference.size(); ++i)
                            expected_grid[i] = next(reference[i], expected[i], p.rule);
                        const int active = int(std::count(work.begin(), work.end(), true));
                        if (active > tiles / 2) full = true;
                        full ? ++dense_steps : ++sparse_steps;
                        std::vector<bool> next_work(tiles);
                        for (int tile = 0; tile < tiles; ++tile) if (full || work[tile])
                            for (int strip = 0; strip < sparse_moore_warp::strips_per_tile; ++strip) {
                                const auto summary = emulate_strip(buffers[gen & 1], buffers[(gen + 1) & 1], p, tile, strip, expected);
                                if (!full) {
                                    enqueue(next_work, summary, p, tile, tight != 0);
                                    cleanup_steps += (summary.flags & 1) && !(summary.flags & 2);
                                }
                            }
                        for (std::size_t i = 0; i < reference.size(); ++i) {
                            require(buffers[(gen + 1) & 1][i] == expected_grid[i], "multi-generation sparse cleanup");
                            ++state_checks;
                        }
                        reference.swap(expected_grid); work.swap(next_work);
                    }
                }
}

int main() {
    std::mt19937 rng(20261029);
    arithmetic(rng); evolution(rng);
    require(sparse_steps && dense_steps && cleanup_steps, "exercise sparse, dense fallback and old-input cleanup");
    std::printf("PASS %llu packed-window counts, %llu direct oracle checks, %llu cell-generation states; "
                "R2..32 C2..16 both boundaries/M, partial tiles, strip ownership, %u sparse/%u dense/%u cleanup steps (CPU only)\n",
                count_checks, direct_checks, state_checks, sparse_steps, dense_steps, cleanup_steps);
}
