// CPU reference/orchestration test. Production bit-plane helpers, tile
// geometry and lane ownership are shared; actual CUDA shuffles, atomics and
// inter-kernel ordering require the separate GPU/reference gates.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "../r1_multistate.hpp"
#include "../tile_geometry.hpp"
#include "../r1_sparse_wide.hpp"

static unsigned long long comparisons = 0;
static unsigned sparse_steps = 0, dense_steps = 0, cleanup_events = 0, fallback_events = 0;

static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL sparse wide bit-plane CPU test: %s\n", what); std::exit(1); }
}

static unsigned scalar_next(unsigned state, unsigned count, const Rule &r) {
    if (state == 0) return unsigned(r.birth_lo) <= count && count <= unsigned(r.birth_hi);
    if (state == 1)
        return unsigned(r.survive_lo) <= count && count <= unsigned(r.survive_hi)
             ? 1 : 2 % unsigned(r.states);
    return (state + 1) % unsigned(r.states);
}

static std::vector<uint8_t> reference(const std::vector<uint8_t> &src, const Problem &p) {
    std::vector<uint8_t> dst(src.size());
    for (int y = 0; y < p.height; ++y)
    for (int x = 0; x < p.width; ++x) {
        unsigned count = 0;
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (!p.rule.count_centre && dx == 0 && dy == 0) continue;
            if (p.rule.shape == Shape::VonNeumann && std::abs(dx) + std::abs(dy) > 1) continue;
            int nx = x + dx, ny = y + dy;
            if (p.boundary == Boundary::Torus) {
                nx = (nx + p.width) % p.width;
                ny = (ny + p.height) % p.height;
            } else if (nx < 0 || nx >= p.width || ny < 0 || ny >= p.height) continue;
            count += src[std::size_t(ny) * p.width + nx] == 1;
        }
        const std::size_t i = std::size_t(y) * p.width + x;
        dst[i] = uint8_t(scalar_next(src[i], count, p.rule));
    }
    return dst;
}

template <int Planes>
static std::vector<uint32_t> pack(const std::vector<uint8_t> &src, const Problem &p) {
    const int pitch = (p.width + 31) / 32;
    const std::size_t words = r1_multistate::plane_words(p);
    std::vector<uint32_t> bits(static_cast<std::size_t>(Planes) * words);
    for (int y = 0; y < p.height; ++y)
    for (int x = 0; x < p.width; ++x)
    for (int b = 0; b < Planes; ++b)
        bits[std::size_t(b) * words + std::size_t(y) * pitch + x / 32]
            |= ((unsigned(src[std::size_t(y) * p.width + x]) >> b) & 1u) << (x & 31);
    return bits;
}

template <int Planes>
static std::vector<uint8_t> unpack(const std::vector<uint32_t> &src, const Problem &p) {
    const int pitch = (p.width + 31) / 32;
    const std::size_t words = r1_multistate::plane_words(p);
    std::vector<uint8_t> out(std::size_t(p.width) * p.height);
    for (int y = 0; y < p.height; ++y)
    for (int wx = 0; wx < pitch; ++wx) {
        r1_multistate::Word<Planes> word;
        for (int b = 0; b < Planes; ++b)
            word.bit[b] = src[std::size_t(b) * words + std::size_t(y) * pitch + wx];
        for (int lane = 0; lane < 32 && wx * 32 + lane < p.width; ++lane)
            out[std::size_t(y) * p.width + wx * 32 + lane]
                = uint8_t(r1_multistate::extract(word, unsigned(lane)));
    }
    return out;
}

template <int Planes>
static void run(Problem p, const std::vector<uint8_t> &initial, bool tight) {
    const int tile_w = r1_sparse_wide::width, tile_h = r1_sparse_wide::height;
    const int tx = (p.width + tile_w - 1) / tile_w, ty = (p.height + tile_h - 1) / tile_h;
    const int tiles = tx * ty, pitch = (p.width + 31) / 32;
    const std::size_t words = r1_multistate::plane_words(p);
    const bool torus = p.boundary == Boundary::Torus;
    std::vector<uint8_t> nz(std::size_t(tiles), 0);
    for (int y = 0; y < p.height; ++y)
    for (int x = 0; x < p.width; ++x)
        nz[std::size_t(y / tile_h) * tx + x / tile_w] |= initial[std::size_t(y) * p.width + x] != 0;
    std::vector<int> work;
    // Same initial byte-grid support inspection as SparseWorkspace::initialize.
    for (int tile = 0; tile < tiles; ++tile) {
        const auto xs = tile_geometry::influence(tile % tx, tile_w, p.width, 1, torus);
        const auto ys = tile_geometry::influence(tile / tx, tile_h, p.height, 1, torus);
        bool needed = false;
        for (int y = 0; y < ys.count; ++y)
        for (int x = 0; x < xs.count; ++x) needed |= nz[std::size_t(ys.ids[y]) * tx + xs.ids[x]] != 0;
        if (needed) work.push_back(tile);
    }
    std::vector<uint32_t> buffer[2];
    buffer[0] = pack<Planes>(initial, p);
    buffer[1].assign(buffer[0].size(), 0); // pack -> swap -> clear opposite packed prefix
    auto expected = initial;
    int parity = 0;
    bool full = false;
    bool previously_sparse = false;
    for (int t = 0; t < p.steps; ++t) {
        if (!full && int(work.size()) > tiles / 2) {
            full = true;
            fallback_events += previously_sparse;
        }
        if (full) ++dense_steps;
        else { ++sparse_steps; previously_sparse = true; }
        const auto &src = buffer[parity];
        auto &dst = buffer[parity ^ 1];
        std::vector<int> next;
        std::vector<uint8_t> stamped(std::size_t(tiles), 0);
        auto enqueue = [&](int tile) {
            require(tile >= 0 && tile < tiles, "queue tile in range");
            if (!stamped[tile]) { stamped[tile] = 1; next.push_back(tile); }
        };
        auto enqueue_influence = [&](int begin_x, int end_x, int begin_y, int end_y) {
            const auto xs = tile_geometry::influence_interval(begin_x, end_x, tile_w, p.width, 1, torus);
            const auto ys = tile_geometry::influence_interval(begin_y, end_y, tile_h, p.height, 1, torus);
            for (int lane = 0; lane < 32; ++lane) {
                const int id = r1_sparse_wide::candidate(lane, xs, ys, tx, -1);
                if (id >= 0) enqueue(id);
            }
        };
        auto load = [&](int wx, int y) {
            r1_multistate::Word<Planes> result;
            if (wx < 0 || wx >= pitch) return result;
            if (torus) y = (y % p.height + p.height) % p.height;
            else if (y < 0 || y >= p.height) return result;
            for (int b = 0; b < Planes; ++b)
                result.bit[b] = src[std::size_t(b) * words + std::size_t(y) * pitch + wx];
            return result;
        };
        auto row = [&](int wx, int y) {
            const int prev = torus && wx == 0 ? pitch - 1 : wx - 1;
            const int following = torus && wx == pitch - 1 ? 0 : wx + 1;
            return r1_multistate::horizontal(
                r1_multistate::alive(load(wx, y), p.rule.states),
                r1_multistate::alive(load(prev, y), p.rule.states),
                r1_multistate::alive(load(following, y), p.rule.states),
                wx, p.width, torus);
        };
        const int count = full ? tiles : int(work.size());
        for (int item = 0; item < count; ++item) {
            const int tile = full ? item : work[item];
            const int x0 = tile % tx * tile_w, y0 = tile / tx * tile_h;
            r1_sparse_wide::Bounds bounds;
            unsigned flags = 0;
            // Use the exact lane/row ownership map from the CUDA header.
            // Neighbor words are loaded explicitly here: this emulates the
            // arithmetic, not CUDA's width=8 subgroup-shuffle mechanism.
            for (int lane = 0; lane < 32; ++lane)
            for (int r = 0; r < r1_sparse_wide::rows_per_lane; ++r) {
                const int wx = x0 / 32 + r1_sparse_wide::column(lane);
                const int y = y0 + r1_sparse_wide::first_row(lane) + r;
                if (wx >= pitch || y >= p.height) continue;
                const auto up = row(wx, y - 1), mid = row(wx, y), down = row(wx, y + 1);
                const auto n = p.rule.shape == Shape::Moore
                    ? r1_bits::count9(up.left, up.centre, up.right, mid.left, mid.right,
                        down.left, down.centre, down.right, p.rule.count_centre ? mid.centre : 0)
                    : r1_bits::count9(0, up.centre, 0, mid.left, mid.right, 0,
                        down.centre, 0, p.rule.count_centre ? mid.centre : 0);
                const auto input = load(wx, y);
                const auto output = r1_multistate::next(input, n, p.rule);
                const uint32_t mask = r1_bits::mask(p.width, wx);
                uint32_t before = 0, after = 0;
                for (int b = 0; b < Planes; ++b) {
                    dst[std::size_t(b) * words + std::size_t(y) * pitch + wx] = output.bit[b] & mask;
                    before |= input.bit[b] & mask;
                    after |= output.bit[b] & mask;
                }
                flags |= unsigned(before != 0) | (unsigned(after != 0) << 1);
                const uint32_t live = r1_multistate::alive(output, p.rule.states) & mask;
                for (int bit = 0; bit < 32; ++bit)
                    if ((live >> bit) & 1u) bounds.include(wx * 32 - x0 + bit, y - y0);
            }
            if (!full) {
                cleanup_events += (flags & 3) == 1;
                if (tight) {
                    if (!bounds.empty()) enqueue_influence(x0 + bounds.xmin, x0 + bounds.xmax + 1,
                                                          y0 + bounds.ymin, y0 + bounds.ymax + 1);
                    if (flags & 3) enqueue(tile); // old input is required, even with zero output
                } else {
                    if (flags & 2) enqueue_influence(x0, std::min(x0 + tile_w, p.width),
                                                    y0, std::min(y0 + tile_h, p.height));
                    if (flags & 1) enqueue(tile);
                }
            }
        }
        parity ^= 1;
        work.swap(next);
        expected = reference(expected, p);
        require(unpack<Planes>(buffer[parity], p) == expected,
                "sparse queue / moving support / decay / dense fallback differs from reference");
        comparisons += expected.size();
        for (int b = 0; b < Planes; ++b)
        for (int y = 0; y < p.height; ++y)
            require((buffer[parity][std::size_t(b) * words + std::size_t(y) * pitch + pitch - 1]
                     & ~r1_bits::mask(p.width, pitch - 1)) == 0, "tail padding stays zero");
    }
}

static void dispatch(const Problem &p, const std::vector<uint8_t> &grid, bool tight) {
    switch (r1_multistate::plane_count(p.rule.states)) {
        case 1: run<1>(p, grid, tight); break;
        case 2: run<2>(p, grid, tight); break;
        case 3: run<3>(p, grid, tight); break;
        default: run<4>(p, grid, tight); break;
    }
}

int main() {
    unsigned ownership[r1_sparse_wide::height * r1_sparse_wide::words_per_row]{};
    for (int lane = 0; lane < 32; ++lane) {
        require(r1_sparse_wide::first_row(lane) == r1_sparse_wide::first_row(lane ^ 7),
                "eight-lane shuffle group shares rows");
        for (int row = 0; row < r1_sparse_wide::rows_per_lane; ++row)
            ++ownership[(r1_sparse_wide::first_row(lane) + row) * r1_sparse_wide::words_per_row
                        + r1_sparse_wide::column(lane)];
    }
    for (unsigned count : ownership) require(count == 1, "each 256x16 output word has one writer");
    for (int nx = 0; nx <= 4; ++nx)
    for (int ny = 0; ny <= 4; ++ny)
    for (int self : {-1, 95}) {
        tile_geometry::AxisTiles xs, ys;
        for (int x = 0; x < nx; ++x) xs.append(x * 2);
        for (int y = 0; y < ny; ++y) ys.append(y * 2);
        unsigned candidates = 0;
        for (int lane = 0; lane < 32; ++lane) {
            const int id = r1_sparse_wide::candidate(lane, xs, ys, 13, self);
            candidates += id >= 0;
            if (lane < nx * ny)
                require(id == ys.ids[lane / nx] * 13 + xs.ids[lane % nx], "candidate Cartesian product");
            else require(id == (lane == nx * ny ? self : -1), "cleanup candidate and inactive lanes");
        }
        require(candidates == unsigned(nx * ny + (self >= 0)), "all candidates fit one warp");
    }
    std::mt19937 rng(20261029);
    const int widths[] = {3, 33, 257, 513, 1025};
    const int heights[] = {3, 17, 33, 49, 67};
    const int state_counts[] = {2, 3, 4, 5, 8, 9, 16};
    unsigned cases = 0;
    for (int states : state_counts)
    for (int size = 0; size < 5; ++size)
    for (Shape shape : {Shape::Moore, Shape::VonNeumann})
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
    for (bool centre : {false, true})
    for (bool tight : {false, true}) {
        const int maxn = (shape == Shape::Moore ? 8 : 4) + int(centre);
        const int slo = int(centre) + int(rng() % unsigned(maxn - int(centre) + 1));
        const int shi = slo + int(rng() % unsigned(maxn - slo + 1));
        const int blo = 1 + int(rng() % unsigned(maxn));
        const int bhi = blo + int(rng() % unsigned(maxn - blo + 1));
        const Problem p{widths[size], heights[size],
                        {1, states, centre, slo, shi, blo, bhi, shape}, boundary, 5};
        std::vector<uint8_t> grid(std::size_t(p.width) * p.height);
        for (int sample = 0; sample < 20; ++sample) {
            const int x = int(rng() % unsigned(p.width)), y = int(rng() % unsigned(p.height));
            grid[std::size_t(y) * p.width + x] = uint8_t(1 + rng() % unsigned(states - 1));
        }
        grid.front() = 1;
        grid.back() = uint8_t(states - 1);
        dispatch(p, grid, tight);
        ++cases;
    }
    // Broad empty regions keep these runs sparse for many generations. Both
    // tile edges and a partial final tile/word are crossed by active support.
    for (int states : {2, 3})
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
    for (bool tight : {false, true}) {
        const Rule rule = states == 2
            ? Rule{1, 2, false, 2, 3, 3, 3, Shape::Moore}
            : Rule{1, 3, false, 8, 8, 2, 2, Shape::Moore};
        const Problem p{1027, 257, rule, boundary, 23};
        std::vector<uint8_t> grid(std::size_t(p.width) * p.height);
        const int xy[][2] = {{255, 14}, {256, 15}, {254, 16}, {255, 16}, {256, 16},
                             {1025, 255}, {1026, 256}, {1024, 256}};
        for (const auto &cell : xy) grid[std::size_t(cell[1]) * p.width + cell[0]] = 1;
        if (states == 3) grid[100 * p.width + 767] = 2;
        dispatch(p, grid, tight);
        ++cases;
    }
    for (bool tight : {false, true}) {
        // A single last-phase cell dies at generation one. Its old buffer must
        // still be queued and cleared at generation two, without any births.
        for (int states : {3, 4, 16}) {
            const Problem p{321, 161, {1, states, false, 2, 3, 3, 3, Shape::Moore}, Boundary::Dead, 5};
            std::vector<uint8_t> grid(std::size_t(p.width) * p.height);
            grid[20 * p.width + 20] = uint8_t(states - 1);
            dispatch(p, grid, tight);
            ++cases;
        }
        // Gliders cross x=256/y=16 tile edges and a partial-width torus seam.
        for (int origin : {14, 95}) {
            const Problem p{641, 97, {1, 2, false, 2, 3, 3, 3, Shape::Moore}, Boundary::Torus, 40};
            std::vector<uint8_t> grid(std::size_t(p.width) * p.height);
            const int x0 = origin == 14 ? 254 : 639, y0 = origin;
            const int xy[][2] = {{1, 0}, {2, 1}, {0, 2}, {1, 2}, {2, 2}};
            for (const auto &cell : xy)
                grid[std::size_t((y0 + cell[1]) % p.height) * p.width + (x0 + cell[0]) % p.width] = 1;
            dispatch(p, grid, tight);
            ++cases;
        }
        // Monotone B1/S0..8 growth starts sparse, then exceeds the one-way
        // fallback threshold. Subsequent full sweeps must ignore stale queues.
        const Problem p{321, 161, {1, 2, false, 0, 8, 1, 8, Shape::Moore}, Boundary::Torus, 110};
        std::vector<uint8_t> grid(std::size_t(p.width) * p.height);
        grid[80 * p.width + 160] = 1;
        dispatch(p, grid, tight);
        ++cases;
    }
    require(sparse_steps > 0 && dense_steps > 0, "both scheduling paths exercised");
    require(cleanup_events > 0, "old-input cleanup path exercised");
    require(fallback_events > 0, "growth enters dense fallback after sparse execution");
    std::printf("PASS %u cases, %llu cell-generation comparisons; sparse=%u dense=%u old-input-cleanups=%u growth-fallbacks=%u (CPU orchestration only)\n",
                cases, comparisons, sparse_steps, dense_steps, cleanup_events, fallback_events);
}

