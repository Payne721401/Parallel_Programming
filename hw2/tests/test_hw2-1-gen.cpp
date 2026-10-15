// FastGen against the official generator, bit for bit, without MPI calls: every scale 1..26,
// both scramble settings, seeds at the uint64 edges, both AVX-512 level loops (PP_GEN8=0/1)
// and the scalar tail. Build from hw2/tests:
//   mpicxx -std=c++17 -O3 -march=native test_hw2-1-gen.cpp ../hw2-1/graph_gen.cpp -o test-gen
#include "../hw2-1/hw2-1.cpp"

int main()
{
    u64 checked = 0;
    for (int scale = 1; scale <= 26; scale++)
        for (int scramble = 0; scramble < 2; scramble++)
            for (u64 seed : {u64(0), u64(1), u64(2026), u64(0x9e3779b97f4a7c15ULL), ~u64(0)}) {
                const GraphParams p{scale, scale <= 16 ? 256 : 16, seed, scramble, 0};
                const u64 M = num_edges(p), n = std::min<u64>(M, 1003);   // 1003: a scalar tail
                std::vector<u32> u0(n), v0(n), u1(n), v1(n);
                std::vector<u8> w0(n), w1(n);
                for (u64 i0 : {u64(0), M / 2, M - n}) {
                    if (i0 + n > M) continue;
                    generate_edges(p, i0, i0 + n, u0.data(), v0.data(), w0.data());
                    for (bool bytes8 : {false, true}) {
                        const FastGen fg(p, bytes8);
                        fg.edges(i0, n, u1.data(), v1.data(), w1.data());
                        if (u0 != u1 || v0 != v1 || w0 != w1) {
                            std::fprintf(stderr, "FAIL scale %d scramble %d seed %llu start %llu gen8 %d\n", scale,
                                         scramble, (unsigned long long)seed, (unsigned long long)i0, bytes8 ? 1 : 0);
                            return 1;
                        }
                        checked += n;
                    }
                }
            }
#ifdef HW21_AVX512
    const int vector_path = 1;
#else
    const int vector_path = 0;
#endif
    std::printf("PASS FastGen matches the official generator on %llu edges (avx512=%d, both level loops)\n",
                (unsigned long long)checked, vector_path);
}
