// CPU tests execute the exact packed arithmetic shared with the CUDA kernel.
// GPU loads, barriers, stores and worklist scheduling still need Nano5 tests.
#include <cstdio>
#include <cstdlib>
#include <random>
#include "../r1_packed.hpp"

static unsigned byte(uint32_t word, int b) { return (word >> (8*b)) & 255u; }
static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL packed R1: %s\n", what); std::exit(1); }
}

static unsigned scalar_next(unsigned state, unsigned n, const Rule &r) {
    if (state == 0) return unsigned(r.birth_lo) <= n && n <= unsigned(r.birth_hi) ? 1 : 0;
    if (state == 1) return unsigned(r.survive_lo) <= n && n <= unsigned(r.survive_hi) ? 1 : 2 % r.states;
    return (state + 1) % r.states;
}

int main() {
    unsigned long long comparisons = 0;
    // Every combination of four legal state bytes, against every state value.
    for (unsigned bits = 0; bits < 65536; ++bits) {
        uint32_t word = 0;
        for (int b = 0; b < 4; ++b) word |= ((bits >> (4*b)) & 15u) << (8*b);
        for (unsigned target = 0; target < 16; ++target) {
            const uint32_t eq = r1_packed::equal(word, target);
            for (int b = 0; b < 4; ++b) {
                require(byte(eq,b) == unsigned(byte(word,b) == target), "byte equality/borrow");
                ++comparisons;
            }
        }
    }
    // Exhaust every possible four-byte R1 count combination, all intervals.
    for (unsigned code = 0; code < 10000; ++code) {
        uint32_t counts = 0;
        unsigned rest = code;
        for (int b = 0; b < 4; ++b) { counts |= (rest % 10) << (8*b); rest /= 10; }
        for (int lo = 0; lo <= 9; ++lo)
        for (int hi = lo; hi <= 9; ++hi) {
            const uint32_t actual = r1_packed::in_range(counts,lo,hi);
            for (int b = 0; b < 4; ++b) {
                const unsigned n = byte(counts,b);
                require(byte(actual,b) == unsigned(unsigned(lo) <= n && n <= unsigned(hi)), "range/borrow");
                ++comparisons;
            }
        }
    }
    // Every C, survival/birth interval, state, and possible R1 count. Adjacent
    // bytes deliberately differ to detect accidental carry/borrow coupling.
    for (int states = 2; states <= 16; ++states)
    for (int slo = 0; slo <= 9; ++slo)
    for (int shi = slo; shi <= 9; ++shi)
    for (int blo = 1; blo <= 9; ++blo)
    for (int bhi = blo; bhi <= 9; ++bhi) {
        Rule r{1,states,true,slo,shi,blo,bhi,Shape::Moore};
        for (int s = 0; s < states; ++s)
        for (int n = 0; n <= 9; ++n) {
            uint32_t input = 0, counts = 0;
            for (int b = 0; b < 4; ++b) {
                input |= uint32_t((s+3*b)%states) << (8*b);
                counts |= uint32_t((n+7*b)%10) << (8*b);
            }
            const uint32_t actual = r1_packed::next(input,counts,r);
            for (int b = 0; b < 4; ++b) {
                require(byte(actual,b) == scalar_next(byte(input,b),byte(counts,b),r), "state transition");
                ++comparisons;
            }
        }
    }
    std::mt19937 rng(20261023);
    // Check both overlapping output rows against independent scalar counts.
    // Randomize states, not just alive bits, so decaying cells cannot count.
    for (int sample = 0; sample < 20000; ++sample) {
        const int states = 2 + sample % 15;
        unsigned cells[4][12];
        uint32_t words[12]{};
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 12; ++x) {
                cells[y][x] = rng() % states;
                words[3*y+x/4] |= unsigned(cells[y][x] == 1) << (8*(x%4));
            }
        for (bool moore : {false,true})
        for (bool centre : {false,true}) {
            const auto pair = moore
                ? r1_packed::count_pair<true>(words[0],words[1],words[2],words[3],words[4],words[5],words[6],words[7],words[8],words[9],words[10],words[11],centre)
                : r1_packed::count_pair<false>(words[0],words[1],words[2],words[3],words[4],words[5],words[6],words[7],words[8],words[9],words[10],words[11],centre);
            for (int row = 0; row < 2; ++row)
            for (int b = 0; b < 4; ++b) {
                unsigned expected = 0;
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!centre && !dx && !dy) continue;
                    if (!moore && std::abs(dx)+std::abs(dy)>1) continue;
                    expected += cells[row+1+dy][4+b+dx] == 1;
                }
                require(byte(row == 0 ? pair.first : pair.second,b) == expected,"shared pair counts");
                ++comparisons;
            }
        }
    }
    for (int sample = 0; sample < 20000; ++sample) {
        unsigned cells[3][12];
        uint32_t words[9]{};
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 12; ++x) {
                cells[y][x] = rng() & 1u;
                words[3*y+x/4] |= cells[y][x] << (8*(x%4));
            }
        for (bool moore : {false,true})
        for (bool centre : {false,true}) {
            const uint32_t actual = moore
                ? r1_packed::count<true>(words[0],words[1],words[2],words[3],words[4],words[5],words[6],words[7],words[8],centre)
                : r1_packed::count<false>(words[0],words[1],words[2],words[3],words[4],words[5],words[6],words[7],words[8],centre);
            for (int b = 0; b < 4; ++b) {
                unsigned expected = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!centre && !dx && !dy) continue;
                        if (!moore && std::abs(dx)+std::abs(dy) > 1) continue;
                        expected += cells[1+dy][4+b+dx];
                    }
                require(byte(actual,b) == expected, "packed neighbour shifts/sums");
                ++comparisons;
            }
        }
    }
    std::printf("PASS %llu scalar comparisons of production packed R1 arithmetic (CPU only)\n", comparisons);
}
