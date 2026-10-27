// Test-only executable. Reuse one CUDA context and compare every output cell.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "ca.hpp"

// The unmodified course reference is compiled only into this test executable.
#define run run_reference
#include "../original_version/hw3-1.cu"
#undef run

static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

static int tested = 0;
static void compare(const Problem &p, const std::vector<uint8_t> &initial) {
    std::vector<uint8_t> expected = initial;
    run_reference(expected.data(), p);
    uint8_t *actual = nullptr;
    check(cudaMallocHost(&actual, initial.size()));
    std::copy(initial.begin(), initial.end(), actual);
    run(actual, p);
    check(cudaDeviceSynchronize());
    check(cudaGetLastError());
    for (size_t i = 0; i < initial.size(); ++i) {
        if (actual[i] != expected[i]) {
            const Rule &r = p.rule;
            std::fprintf(stderr,
                "FAIL case=%d W=%d H=%d T=%d R=%d C=%d M=%d shape=%d boundary=%d "
                "S=%d..%d B=%d..%d at=(%zu,%zu) expected=%u actual=%u\n",
                tested, p.width, p.height, p.steps, r.radius, r.states,
                int(r.count_centre), int(r.shape), int(p.boundary),
                r.survive_lo, r.survive_hi, r.birth_lo, r.birth_hi,
                i % p.width, i / p.width, unsigned(expected[i]), unsigned(actual[i]));
            std::exit(1);
        }
    }
    check(cudaFreeHost(actual));
    ++tested;
}

static void sparse_checks(std::mt19937 &rng) {
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus}) {
        Rule life{1, 2, false, 2, 3, 3, 3, Shape::Moore};
        Problem p{321, 161, life, boundary, 7}; // 6x6 tiles, with thin final tiles
        std::vector<uint8_t> initial(size_t(p.width)*p.height, 0);
        auto put = [&](int x, int y, uint8_t s) {
            initial[size_t(y % p.height)*p.width + x % p.width] = s;
        };
        compare(p, initial); // empty worklist, no generation work necessary
        put(320, 160, 1);
        p.rule = Rule{1, 2, false, 8, 8, 8, 8, Shape::Moore};
        p.steps = 5;
        compare(p, initial); // death must clear the reused source buffer
        p.rule = life;
        std::fill(initial.begin(), initial.end(), 0);
        const int origins[][2] = {{62, 30}, {319, 159}};
        for (const auto &origin : origins) {
            const int x = origin[0], y = origin[1];
            put(x+1,y,1); put(x+2,y+1,1);
            put(x,y+2,1); put(x+1,y+2,1); put(x+2,y+2,1);
        }
        for (int steps : {0, 1, 2, 9, 16}) {
            p.steps = steps;
            compare(p, initial); // motion across tile boundaries and torus seams
        }
        p.rule = Rule{1, 3, false, 8, 8, 2, 2, Shape::Moore};
        p.steps = 9;
        for (int y = 27; y < 36; ++y)
            for (int x = 59; x < 70; ++x) put(x,y,uint8_t(rng()%3));
        put(320,160,2);
        compare(p, initial); // C3, including decay initially and after live cells die
        p.rule = Rule{1, 16, false, 0, 8, 1, 8, Shape::Moore};
        std::fill(initial.begin(), initial.end(), 0);
        put(0,0,2); put(63,31,7); put(64,32,15); put(320,160,2);
        p.steps = 17;
        compare(p, initial); // no alive cells: nonzero tracking must continue decay
        p.rule = Rule{1, 2, false, 0, 8, 1, 8, Shape::Moore};
        std::fill(initial.begin(), initial.end(), 0);
        for (int x = 0; x < p.width; ++x) put(x,31,1);
        p.steps = 70;
        compare(p, initial); // initially sparse band grows past dense fallback threshold
        p.rule = life;
        p.steps = 3;
        for (auto &s : initial) s = uint8_t(rng()%2);
        compare(p, initial); // forced entry with a full queue; dense fallback immediately
    }
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
    for (Shape shape : {Shape::Moore, Shape::VonNeumann})
    for (bool centre : {false, true}) {
        Rule r{5, shape == Shape::Moore ? 2 : 4, centre, 12, 34, 34, 52, shape};
        Problem p{321, 161, r, boundary, 4};
        std::vector<uint8_t> initial(size_t(p.width)*p.height, 0);
        for (int y = 22; y < 50; ++y)
            for (int x = 54; x < 82; ++x) initial[size_t(y)*p.width+x] = rng()%100 < 40;
        // Last tile width/height is 1: radius 5 reaches beyond its immediate tile neighbour.
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) initial[size_t(p.height-1-y)*p.width+p.width-1-x] = 1;
        compare(p, initial);
    }
    std::printf("PASS %d/%d (sparse full-grid comparison; not timing)\n", tested, tested);
}

static void support_checks() {
    for (int radius : {1,5})
    for (Boundary boundary : {Boundary::Dead,Boundary::Torus})
    for (Shape shape : {Shape::Moore,Shape::VonNeumann})
    for (int pattern=0; pattern<3; ++pattern) {
        const int maxn=shape==Shape::Moore ? (2*radius+1)*(2*radius+1)-1 : 2*radius*(radius+1);
        Rule r{radius,16,pattern==1,0,maxn+int(pattern==1),1,1,shape};
        Problem p{321,161,r,boundary,pattern==2 ? 17 : 4};
        std::vector<uint8_t> initial(size_t(p.width)*p.height,0);
        if (pattern==0) initial[15*p.width+31]=1; // interior support
        if (pattern==1) {
            initial[31*p.width+63]=1; // output crosses both tile boundaries
            initial[160*p.width+320]=1; // one-cell tail and torus seam
        }
        initial[30*p.width+62]=2;
        initial[159*p.width+319]=15; // decay must not expand alive bounds
        compare(p,initial);
    }
    for (Boundary boundary : {Boundary::Dead,Boundary::Torus})
    for (Shape shape : {Shape::Moore,Shape::VonNeumann}) {
        const int maxn=shape==Shape::Moore ? 4224 : 2112;
        Problem p{257,257,Rule{32,4,true,1,maxn+1,maxn,maxn+1,shape},boundary,2};
        std::vector<uint8_t> initial(size_t(p.width)*p.height,0);
        initial[128*p.width+128]=1;
        initial[159*p.width+191]=3;
        compare(p,initial); // R equals tile height; bbox used on second step
    }
    std::printf("PASS %d/%d (alive support, decay, cleanup and large-radius checks; not timing)\n",tested,tested);
}

static void r1_checks(std::mt19937 &rng, bool report = true) {
    const int state_counts[] = {2, 3, 3, 16, 4, 2, 16, 4};
    for (int width : {3, 4, 65, 68, 128, 129, 641, 644})
    for (int mode = 0; mode < 8; ++mode) {
        const bool moore = (mode & 1) == 0;
        const bool torus = (mode & 2) != 0;
        const bool centre = (mode & 4) != 0;
        const int maxn = (moore ? 8 : 4) + int(centre);
        Rule r{1, state_counts[mode], centre, int(centre), maxn, 2, 2,
               moore ? Shape::Moore : Shape::VonNeumann};
        if (moore && r.states == 2 && !centre) {
            r.survive_lo = 2; r.survive_hi = 3; r.birth_lo = 3; r.birth_hi = 3;
        } else if (moore && r.states == 3 && !centre) {
            r.survive_lo = 8; r.survive_hi = 8; // same R1/C3 rule family as public 10
        }
        // Largest grids leave enough empty tiles that the separated patches
        // exercise tracking, rather than immediately taking dense fallback.
        Problem p{width, width < 65 ? 7 : width >= 641 ? 321 : 161, r,
                  torus ? Boundary::Torus : Boundary::Dead, 3 + mode % 3};
        std::vector<uint8_t> initial(size_t(p.width)*p.height, 0);
        // Local patches cross word/tile boundaries; thin tails and both seams
        // exercise scalar packing, vector loads/stores, and valid-byte masking.
        for (int y = 0; y < p.height; ++y)
            for (int x = 0; x < p.width; ++x)
                if ((x < 6 || x >= width-6 || (x >= 60 && x <= 69))
                        && (y < 6 || y >= p.height-6 || (y >= 29 && y <= 35)))
                    initial[size_t(y)*p.width+x] = uint8_t(rng()%r.states);
        compare(p, initial);
    }
    if (report)
        std::printf("PASS %d/%d (R1 packed boundaries and rules; not timing)\n", tested, tested);
}

static void dense_r1_checks(std::mt19937 &rng) {
    r1_checks(rng, false); // 64 aligned/unaligned, Moore/VN, M0/M1, C2..16 cases
    for (int states : {2, 4}) {
        Rule r{1, states, false, states == 2 ? 4 : 3, 5, 2, 2, Shape::Moore};
        for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
        for (int pattern = 0; pattern < 5; ++pattern) {
            Problem p{129, 65, r, boundary, 17 + pattern};
            std::vector<uint8_t> initial(size_t(p.width) * p.height, 0);
            if (pattern == 1 || pattern == 2) {
                // Nine live cells: same density, either one cluster or one per tile.
                for (int y = 0; y < 3; ++y)
                    for (int x = 0; x < 3; ++x) {
                        const int gx = pattern == 1 ? 62 + x : 64 * x;
                        const int gy = pattern == 1 ? 30 + y : 32 * y;
                        initial[size_t(gy) * p.width + gx] = 1;
                    }
            } else if (pattern == 3) {
                for (auto &s : initial) s = uint8_t(rng() % states);
            } else if (pattern == 4) {
                // Dense initially, then empty: decay without births (or C2 death).
                for (auto &s : initial)
                    s = states == 2 ? 1 : uint8_t(2 + rng() % (states - 2));
                if (states == 2) p.rule.survive_lo = p.rule.survive_hi = 0;
            }
            compare(p, initial); // includes empty and thin final tiles
        }
        for (int steps : {0, 403, 1190}) {
            Problem p{17, 19, r, states == 2 ? Boundary::Torus : Boundary::Dead, steps};
            std::vector<uint8_t> initial(size_t(p.width) * p.height);
            for (auto &s : initial) s = uint8_t(rng() % states);
            compare(p, initial); // zero steps and long odd/even ping-pong chains
        }
    }
    std::printf("PASS %d/%d (dense R1 boundaries, distributions and long runs; not timing)\n", tested, tested);
}

static void sliding_checks(std::mt19937 &rng) {
    for (int radius : {1, 2, 5, 12, 20, 32})
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
    for (bool centre : {false, true})
    for (int variant = 0; variant < 2; ++variant) {
        const int maxn = (2 * radius + 1) * (2 * radius + 1) - 1 + int(centre);
        Rule r{radius, variant == 0 ? 2 : centre ? 16 : 4, centre,
               variant == 0 ? int(centre) : maxn / 3,
               variant == 0 ? maxn : maxn / 2,
               variant == 0 ? maxn : maxn / 4,
               variant == 0 ? maxn : maxn / 3, Shape::Moore};
        const int side = 2 * radius + 1;
        Problem p{side + 7 * variant, side + 3 * variant, r, boundary, 1 + variant};
        std::vector<uint8_t> initial(size_t(p.width) * p.height);
        for (auto &s : initial)
            s = rng() % 100 < 35 ? 1 : uint8_t(rng() % r.states);
        // Minimum dimensions wrap the tile halo repeatedly; larger odd/partial
        // dimensions cross output segments and tiles. Compare every cell.
        compare(p, initial);
    }
    std::printf("PASS %d/%d (Moore sliding radius/boundary/state checks; not timing)\n", tested, tested);
}

static void graph_checks(std::mt19937 &rng) {
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus}) {
        Rule life{1, 2, false, 2, 3, 3, 3, Shape::Moore};
        for (int steps : {0, 1, 31, 32, 33, 63, 64, 65, 97}) {
            Problem p{steps % 2 ? 321 : 320, 161, life, boundary, steps};
            std::vector<uint8_t> initial(size_t(p.width) * p.height, 0);
            auto put = [&](int x, int y) {
                initial[size_t(y % p.height) * p.width + x % p.width] = 1;
            };
            // Moving gliders cross tiles/seams and revisit stamp storage after replay.
            const int origins[][2] = {{62, 30}, {p.width-2, 159}};
            for (const auto &origin : origins) {
                int x = origin[0], y = origin[1];
                put(x+1,y); put(x+2,y+1); put(x,y+2); put(x+1,y+2); put(x+2,y+2);
            }
            compare(p, initial);
        }
        for (int pattern = 0; pattern < 3; ++pattern) {
            Problem p{321, 161, life, boundary, 97};
            std::vector<uint8_t> initial(size_t(p.width) * p.height, 0);
            if (pattern == 1) {
                p.rule.states = 16;
                initial[0] = 2; initial[63] = 15; // Decay then empty across replays.
            } else if (pattern == 2) {
                p.rule = Rule{1, 2, false, 0, 8, 1, 8, Shape::Moore};
                for (int x = 0; x < p.width; ++x) initial[31*p.width+x] = 1;
                // Sparse growth triggers one-way dense fallback during the graph.
            }
            compare(p, initial);
        }
        for (int radius : {1, 2, 5})
        for (Shape shape : {Shape::Moore, Shape::VonNeumann}) {
            const int maxn = shape == Shape::Moore ? (2*radius+1)*(2*radius+1) : 2*radius*(radius+1)+1;
            Rule r{radius, 4, true, 1, maxn, 2, 3, shape};
            Problem p{33, 17, r, boundary, 65};
            std::vector<uint8_t> initial(size_t(p.width) * p.height);
            for (auto &s : initial) s = uint8_t(rng()%4);
            compare(p, initial); // M1, C4, VN, direct R2 and tiled R5, odd tail.
        }
        Problem p{7, 9, life, boundary, 5000};
        std::vector<uint8_t> initial(63, 0);
        initial[16] = initial[17] = initial[23] = initial[24] = 1;
        compare(p, initial); // Maximum T, repeated run() calls must reset epoch.
    }
    std::printf("PASS %d/%d (graph batches, parity, epochs and fallback; not timing)\n", tested, tested);
}

int main(int argc, char **argv) {
    int random_cases = 64;
    unsigned seed = 20261022;
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    const bool sparse = argc == 2 && std::strcmp(argv[1], "--sparse") == 0;
    const bool r1 = argc == 2 && std::strcmp(argv[1], "--r1") == 0;
    const bool dense_r1 = argc == 2 && std::strcmp(argv[1], "--dense-r1") == 0;
    const bool sliding = argc == 2 && std::strcmp(argv[1], "--sliding") == 0;
    const bool graphs = argc == 2 && std::strcmp(argv[1], "--graphs") == 0;
    const bool support = argc == 2 && std::strcmp(argv[1], "--support") == 0;
    if (argc > 3) {
        std::fprintf(stderr, "usage: %s [random_cases:0..1000] [seed] | --quick | --sparse | --r1 | --dense-r1 | --sliding | --graphs | --support\n", argv[0]);
        return 2;
    }
    if (argc >= 2 && !quick && !sparse && !r1 && !dense_r1 && !sliding && !graphs && !support) random_cases = std::atoi(argv[1]);
    if (argc >= 3) seed = unsigned(std::strtoul(argv[2], nullptr, 10));
    if (random_cases < 0 || random_cases > 1000) return 2;
    const char *kernel = std::getenv("PP_KERNEL");
    const char *active = std::getenv("PP_ACTIVE");
    const char *r1_mode = std::getenv("PP_R1");
    const char *moore_mode = std::getenv("PP_MOORE");
    std::printf("kernel=%s active=%s r1=%s moore=%s seed=%u mode=%s random_cases=%d\n",
                kernel ? kernel : "auto", active ? active : "auto", r1_mode ? r1_mode : "auto",
                moore_mode ? moore_mode : "auto", seed,
                support ? "support" : graphs ? "graphs" : sliding ? "sliding" : dense_r1 ? "dense-r1" : r1 ? "r1" : sparse ? "sparse" : quick ? "quick" : "full",
                (quick || sparse || r1 || dense_r1 || sliding || graphs || support) ? 0 : random_cases);
    std::printf("graph=%s\n", std::getenv("PP_GRAPH") ? std::getenv("PP_GRAPH") : "0");
    std::printf("support=%s\n", std::getenv("PP_SUPPORT") ? std::getenv("PP_SUPPORT") : "default");
    check(cudaFree(nullptr));
    std::mt19937 rng(seed);
    if (support) { support_checks(); return 0; }
    if (graphs) {
        graph_checks(rng);
        return 0;
    }
    if (sliding) {
        sliding_checks(rng);
        return 0;
    }
    if (dense_r1) {
        dense_r1_checks(rng);
        return 0;
    }
    if (r1) {
        r1_checks(rng);
        return 0;
    }
    if (sparse) {
        sparse_checks(rng);
        return 0;
    }

    if (quick) {
        // Twenty deliberately varied cases: multi-tile/partial blocks, narrow
        // torus wrapping, multiple scan chunks, decay, T parity, and R extremes.
        for (int radius : {1, 5, 12, 20, 32})
        for (int mode = 0; mode < 4; ++mode) {
            const bool moore = (mode & 1) == 0;
            const bool torus = (mode & 2) != 0;
            const bool centre = torus;
            const int states = moore ? 2 : 16;
            const int maxn = (moore ? (2*radius+1)*(2*radius+1)-1 : 2*radius*(radius+1)) + int(centre);
            const int typical = maxn * 35 / 100;
            Rule r{radius, states, centre, std::max(int(centre), typical-2), std::min(maxn, typical+3),
                   std::max(1, typical-1), std::min(maxn, typical+1),
                   moore ? Shape::Moore : Shape::VonNeumann};
            Problem p{torus ? 2*radius+1 : 67, radius == 1 ? 35 : 2*radius+3, r,
                      torus ? Boundary::Torus : Boundary::Dead,
                      radius >= 12 ? 1 : (moore ? 3 : 2)};
            std::vector<uint8_t> initial(size_t(p.width) * p.height);
            for (auto &s : initial) {
                const unsigned draw = rng() % 100;
                s = draw < 35 ? 1 : (states > 2 && draw > 70 ? uint8_t(2 + rng()%(states-2)) : 0);
            }
            initial.front() = uint8_t(states-1);
            initial.back() = 1;
            compare(p, initial);
        }
        Rule r{1, 16, false, 0, 8, 1, 8, Shape::Moore};
        Problem p{7, 9, r, Boundary::Torus, 0};
        std::vector<uint8_t> initial(63, 15);
        compare(p, initial); // no launch, no modification
        p.steps = 17;
        compare(p, initial); // all decay to zero, with no live neighbours
        std::printf("PASS %d/%d (quick full-grid comparison; not timing)\n", tested, tested);
        return 0;
    }

    // Cross rule modes with T parity and minimum/non-aligned dimensions.
    for (Shape shape : {Shape::Moore, Shape::VonNeumann})
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
    for (bool centre : {false, true})
    for (int states : {2, 16})
    for (int steps : {0, 1, 2, 5}) {
        const int radius = 1 + tested % 3;
        const int maxn = shape == Shape::Moore ? (2*radius+1)*(2*radius+1)-1 : 2*radius*(radius+1);
        Rule r{radius, states, centre, int(centre), maxn + int(centre), 1, maxn + int(centre), shape};
        Problem p{steps % 2 ? 35 : 2*radius+1, steps % 2 ? 11 : 2*radius+1, r, boundary, steps};
        std::vector<uint8_t> initial(size_t(p.width) * p.height);
        // Valid decay states exercise transitions without waiting C generations.
        for (auto &s : initial) s = uint8_t(rng() % states);
        compare(p, initial);
        std::fill(initial.begin(), initial.end(), uint8_t(0));
        compare(p, initial);
    }

    // Large radii on small grids, never maximum area AND naive R=32.
    for (int radius : {16, 31, 32})
    for (Shape shape : {Shape::Moore, Shape::VonNeumann})
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus}) {
        const int maxn = shape == Shape::Moore ? (2*radius+1)*(2*radius+1)-1 : 2*radius*(radius+1);
        Rule r{radius, 16, true, maxn, maxn+1, 1, 1, shape};
        Problem p{2*radius+1, 2*radius+3, r, boundary, 1};
        std::vector<uint8_t> initial(size_t(p.width) * p.height, 1);
        initial.front() = 0;
        initial.back() = 15;
        compare(p, initial);
    }

    for (int k = 0; k < random_cases; ++k) {
        const int radius = 1 + int(rng() % 5);
        const int states = 2 + int(rng() % 15);
        const bool centre = rng() % 2;
        const Shape shape = rng() % 2 ? Shape::Moore : Shape::VonNeumann;
        const int maxn = (shape == Shape::Moore ? (2*radius+1)*(2*radius+1)-1 : 2*radius*(radius+1)) + int(centre);
        const int slo = int(centre) + int(rng() % (maxn-int(centre)+1));
        const int blo = 1 + int(rng() % maxn);
        Rule r{radius, states, centre, slo, slo + int(rng() % (maxn-slo+1)),
               blo, blo + int(rng() % (maxn-blo+1)), shape};
        Problem p{2*radius+1+int(rng()%35), 2*radius+1+int(rng()%17), r,
                  rng()%2 ? Boundary::Dead : Boundary::Torus, int(rng()%8)};
        std::vector<uint8_t> initial(size_t(p.width) * p.height);
        for (auto &s : initial) s = uint8_t(rng() % (k%2 ? states : 2));
        compare(p, initial);
    }

    // Decay must continue even when there are no live neighbours.
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus}) {
        Rule r{1, 16, false, 0, 8, 1, 8, Shape::Moore};
        Problem p{35, 11, r, boundary, 17};
        std::vector<uint8_t> initial(size_t(p.width) * p.height, 0);
        for (size_t i = 0; i < initial.size(); i += 7) initial[i] = uint8_t(2 + i%14);
        compare(p, initial);
    }
    std::printf("PASS %d/%d (full grid comparison; not a performance measurement)\n", tested, tested);
}
