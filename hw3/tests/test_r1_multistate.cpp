// Executes the same production bit-plane transition, rule predicates, shift
// arithmetic and tail mask on CPU. CUDA launches, loads and shuffles still need
// a separate GPU/reference gate before selecting the new path by default.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "../r1_multistate.hpp"

static unsigned long long comparisons = 0;

static void require(bool ok, const char *what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL multi-state R1: %s\n", what);
        std::exit(1);
    }
}

static unsigned scalar_next(unsigned state, unsigned count, const Rule &r) {
    if (state == 0) return unsigned(r.birth_lo) <= count && count <= unsigned(r.birth_hi);
    if (state == 1)
        return unsigned(r.survive_lo) <= count && count <= unsigned(r.survive_hi)
             ? 1 : 2 % unsigned(r.states);
    return (state + 1) % unsigned(r.states);
}

template <int Planes>
static void check_transitions(int states) {
    // Exhaust the arithmetic domain, including interval/count pairs not used
    // by a particular M/shape. Different cells deliberately carry different
    // states/counts, so a cross-cell carry or missing plane is observable.
    for (int slo = 0; slo <= 9; ++slo)
    for (int shi = slo; shi <= 9; ++shi)
    for (int blo = 1; blo <= 9; ++blo)
    for (int bhi = blo; bhi <= 9; ++bhi) {
        const Rule rule{1, states, true, slo, shi, blo, bhi, Shape::Moore};
        for (int initial = 0; initial < states; ++initial)
        for (int count = 0; count <= 9; ++count) {
            r1_multistate::Word<Planes> input;
            r1_bits::Count n{};
            for (int lane = 0; lane < 32; ++lane) {
                const unsigned s = unsigned((initial + 3 * lane) % states);
                const unsigned c = unsigned((count + 7 * lane) % 10);
                for (int b = 0; b < Planes; ++b)
                    input.bit[b] |= ((s >> b) & 1u) << lane;
                n.b0 |= (c & 1u) << lane;
                n.b1 |= ((c >> 1) & 1u) << lane;
                n.b2 |= ((c >> 2) & 1u) << lane;
                n.b3 |= ((c >> 3) & 1u) << lane;
            }
            const auto actual = r1_multistate::next(input, n, rule);
            const uint32_t alive = r1_multistate::alive(input, states);
            for (int lane = 0; lane < 32; ++lane) {
                const unsigned s = unsigned((initial + 3 * lane) % states);
                const unsigned c = unsigned((count + 7 * lane) % 10);
                require(r1_multistate::extract(input, lane) == s, "binary state encoding");
                require(((alive >> lane) & 1u) == unsigned(s == 1), "alive excludes decay");
                require(r1_multistate::extract(actual, lane) == scalar_next(s, c, rule),
                        "exhaustive C2..16 transition");
                ++comparisons;
            }
        }
    }
}

static std::vector<uint8_t> scalar_step(const std::vector<uint8_t> &src, const Problem &p) {
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
        const std::size_t index = std::size_t(y) * p.width + x;
        dst[index] = uint8_t(scalar_next(src[index], count, p.rule));
    }
    return dst;
}

template <int Planes>
static std::vector<uint32_t> pack_cpu(const std::vector<uint8_t> &src, const Problem &p) {
    const int pitch = (p.width + 31) / 32;
    const std::size_t words = r1_multistate::plane_words(p);
    std::vector<uint32_t> out(static_cast<std::size_t>(Planes) * words);
    for (int y = 0; y < p.height; ++y)
    for (int x = 0; x < p.width; ++x) {
        const unsigned state = src[std::size_t(y) * p.width + x];
        for (int b = 0; b < Planes; ++b)
            out[std::size_t(b) * words + std::size_t(y) * pitch + x / 32]
                |= ((state >> b) & 1u) << (x & 31);
    }
    return out;
}

template <int Planes>
static std::vector<uint8_t> unpack_cpu(const std::vector<uint32_t> &src, const Problem &p) {
    const int pitch = (p.width + 31) / 32;
    const std::size_t words = r1_multistate::plane_words(p);
    std::vector<uint8_t> out(std::size_t(p.width) * p.height);
    for (int y = 0; y < p.height; ++y)
    for (int wx = 0; wx < pitch; ++wx) {
        r1_multistate::Word<Planes> state;
        for (int b = 0; b < Planes; ++b)
            state.bit[b] = src[std::size_t(b) * words + std::size_t(y) * pitch + wx];
        for (int lane = 0; lane < 32 && wx * 32 + lane < p.width; ++lane)
            out[std::size_t(y) * p.width + wx * 32 + lane]
                = uint8_t(r1_multistate::extract(state, lane));
    }
    return out;
}

template <int Planes>
static void check_grid(Problem p, std::mt19937 &rng) {
    const int pitch = (p.width + 31) / 32;
    const std::size_t words = r1_multistate::plane_words(p);
    std::vector<uint8_t> expected(std::size_t(p.width) * p.height);
    for (auto &state : expected) state = uint8_t(rng() % unsigned(p.rule.states));
    // Force seam states, birth candidates, last-phase decay and legal C3 codes.
    expected.front() = 1;
    expected.back() = uint8_t(p.rule.states - 1);
    auto bits = pack_cpu<Planes>(expected, p);
    require(unpack_cpu<Planes>(bits, p) == expected, "pack/unpack odd dimensions");
    require(r1_multistate::allocation_bytes(p) >= bits.size() * sizeof(uint32_t), "bit allocation size");
    require(r1_multistate::allocation_bytes(p) >= expected.size(), "byte allocation size");
    for (int t = 0; t < p.steps; ++t) {
        std::vector<uint32_t> out(bits.size());
        auto load = [&](int wx, int y) {
            r1_multistate::Word<Planes> state;
            if (p.boundary == Boundary::Torus) {
                wx = (wx + pitch) % pitch;
                y = (y + p.height) % p.height;
            } else if (wx < 0 || wx >= pitch || y < 0 || y >= p.height) return state;
            for (int b = 0; b < Planes; ++b)
                state.bit[b] = bits[std::size_t(b) * words + std::size_t(y) * pitch + wx];
            return state;
        };
        auto row = [&](int wx, int y) {
            return r1_multistate::horizontal(
                r1_multistate::alive(load(wx, y), p.rule.states),
                r1_multistate::alive(load(wx - 1, y), p.rule.states),
                r1_multistate::alive(load(wx + 1, y), p.rule.states),
                wx, p.width, p.boundary == Boundary::Torus);
        };
        for (int y = 0; y < p.height; ++y)
        for (int wx = 0; wx < pitch; ++wx) {
            const auto up = row(wx, y - 1), mid = row(wx, y), down = row(wx, y + 1);
            const auto count = p.rule.shape == Shape::Moore
                ? r1_bits::count9(up.left, up.centre, up.right, mid.left, mid.right,
                                 down.left, down.centre, down.right,
                                 p.rule.count_centre ? mid.centre : 0)
                : r1_bits::count9(0, up.centre, 0, mid.left, mid.right, 0,
                                 down.centre, 0, p.rule.count_centre ? mid.centre : 0);
            const auto result = r1_multistate::next(load(wx, y), count, p.rule);
            for (int b = 0; b < Planes; ++b)
                out[std::size_t(b) * words + std::size_t(y) * pitch + wx]
                    = result.bit[b] & r1_bits::mask(p.width, wx);
        }
        bits.swap(out);
        expected = scalar_step(expected, p);
        require(unpack_cpu<Planes>(bits, p) == expected, "multi-generation grid / boundary / tail");
        comparisons += expected.size();
        for (int b = 0; b < Planes; ++b)
        for (int y = 0; y < p.height; ++y)
            require((bits[std::size_t(b) * words + std::size_t(y) * pitch + pitch - 1]
                     & ~r1_bits::mask(p.width, pitch - 1)) == 0, "zero padding after every generation");
    }
}

int main() {
    for (int states = 2; states <= 16; ++states) {
        switch (r1_multistate::plane_count(states)) {
            case 1: check_transitions<1>(states); break;
            case 2: check_transitions<2>(states); break;
            case 3: check_transitions<3>(states); break;
            default: check_transitions<4>(states); break;
        }
    }
    std::mt19937 rng(20261029);
    const int widths[] = {3, 7, 31, 32, 33, 63, 64, 65, 95, 97};
    unsigned grids = 0;
    for (int states = 2; states <= 16; ++states)
    for (int width : widths)
    for (Shape shape : {Shape::Moore, Shape::VonNeumann})
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
    for (bool centre : {false, true}) {
        const int maxn = (shape == Shape::Moore ? 8 : 4) + int(centre);
        const int slo = int(centre) + int(rng() % unsigned(maxn - int(centre) + 1));
        const int shi = slo + int(rng() % unsigned(maxn - slo + 1));
        const int blo = 1 + int(rng() % unsigned(maxn));
        const int bhi = blo + int(rng() % unsigned(maxn - blo + 1));
        const Problem p{width, 3 + int(rng() % 14),
                        {1, states, centre, slo, shi, blo, bhi, shape}, boundary, 7};
        switch (r1_multistate::plane_count(states)) {
            case 1: check_grid<1>(p, rng); break;
            case 2: check_grid<2>(p, rng); break;
            case 3: check_grid<3>(p, rng); break;
            default: check_grid<4>(p, rng); break;
        }
        ++grids;
    }
    std::printf("PASS %llu scalar comparisons; %u multi-generation grids using production R1 bit-plane arithmetic (CPU only)\n",
                comparisons, grids);
}
