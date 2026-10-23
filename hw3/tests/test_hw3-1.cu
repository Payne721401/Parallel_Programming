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

int main(int argc, char **argv) {
    int random_cases = 64;
    unsigned seed = 20261022;
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    if (argc > 3) {
        std::fprintf(stderr, "usage: %s [random_cases:0..1000] [seed] | --quick\n", argv[0]);
        return 2;
    }
    if (argc >= 2 && !quick) random_cases = std::atoi(argv[1]);
    if (argc >= 3) seed = unsigned(std::strtoul(argv[2], nullptr, 10));
    if (random_cases < 0 || random_cases > 1000) return 2;
    const char *kernel = std::getenv("PP_KERNEL");
    std::printf("kernel=%s seed=%u mode=%s random_cases=%d\n",
                kernel ? kernel : "auto", seed, quick ? "quick" : "full", quick ? 0 : random_cases);
    check(cudaFree(nullptr));
    std::mt19937 rng(seed);

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
