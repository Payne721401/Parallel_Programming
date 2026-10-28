// CPU-only verification of the SAME Boolean arithmetic used by the CUDA kernel.
#include "r1_bits.hpp"
#include <array>
#include <cstdio>
#include <random>
#include <cstdlib>
static void require(bool ok) { if (!ok) { std::fputs("FAIL bit arithmetic\n",stderr); std::exit(1); } }
int main() {
    require(!r1_bits::automatic_candidate(16384,16384,127));
    require(r1_bits::automatic_candidate(16384,16384,128));
    require(!r1_bits::automatic_candidate(16384,16383,5000));
    require(r1_bits::automatic_candidate(32768,8192,128));
    require(!r1_bits::automatic_candidate(2048,2048,5000));
    require(r1_bits::automatic_candidate(32768,32768,5000));
    require(!r1_bits::automatic_candidate(32768,32768,0));
    std::mt19937 rng(20261027);
    unsigned long long checks=0;
    // All 512 nine-input neighbourhoods, plus independent random bit positions.
    for (int sample=0; sample<1512; ++sample) {
        std::array<uint32_t,9> a{};
        for (int j=0;j<9;++j) a[j]=sample<512 ? ((sample>>j)&1 ? ~0u : 0u) : rng();
        auto n=r1_bits::count9(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8]);
        for (int b=0;b<32;++b) {
            unsigned sum=0; for (auto v:a) sum+=(v>>b)&1u;
            require((((n.b0>>b)&1u) | (((n.b1>>b)&1u)<<1) |
                     (((n.b2>>b)&1u)<<2) | (((n.b3>>b)&1u)<<3))==sum);
            for (unsigned t=0;t<=15;++t) {
                require(((r1_bits::ge(n,t)>>b)&1u)==unsigned(sum>=t));
                require(((r1_bits::ge_compact(n,t)>>b)&1u)==unsigned(sum>=t)); checks+=2;
            }
        }
        const uint32_t state=rng();
        // Every legal interval up to count nine; independent survival/birth masks.
        for (unsigned lo=0;lo<=9;++lo) for (unsigned hi=lo;hi<=9;++hi) {
            const auto out=r1_bits::next(state,n,lo,hi,1,9);
            const auto out_birth=r1_bits::next(state,n,0,9,lo==0 ? 1 : lo,hi==0 ? 1 : hi);
            require(r1_bits::next_compact(state,n,lo,hi,1,9)==out);
            require(r1_bits::next_compact(state,n,0,9,lo==0 ? 1 : lo,hi==0 ? 1 : hi)==out_birth);
            checks+=2;
            for (int b=0;b<32;++b) {
                unsigned sum=0; for (auto v:a) sum+=(v>>b)&1u;
                const bool live=(state>>b)&1u;
                require(((out>>b)&1u)==unsigned(live ? sum>=lo&&sum<=hi : sum>=1&&sum<=9));
                require(((out_birth>>b)&1u)==unsigned(live || (sum>=(lo?lo:1)&&sum<=(hi?hi:1))));
                checks+=2;
            }
        }
    }
    for (int w=3;w<1100;++w) for (int q=0;q<(w+31)/32+1;++q)
        for (int b=0;b<32;++b) { require(((r1_bits::mask(w,q)>>b)&1u)==unsigned(q*32+b<w)); ++checks; }
    std::printf("PASS %llu bit arithmetic/interval/padding checks (CPU only)\n",checks);
}
