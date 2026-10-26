// Small independent CPU-reference evolution/count oracle. GPU execution: Slurm only.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "ca.hpp"
#include "tile_geometry.hpp"
#define run run_reference
#include "../original_version/hw3-1.cu"
#undef run

static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}

static void expected(int generation, const Problem &p, const std::vector<uint8_t> &now,
                     const std::vector<uint8_t> *old) {
    const int tx = (p.width+63)/64, ty = (p.height+31)/32, count = tx*ty;
    std::vector<int> nz(count), prev(count), diff(count), influence(count);
    unsigned long long alive = 0, nonzero = 0, changed = 0, bins[5]{};
    for (int y = 0; y < p.height; ++y)
        for (int x = 0; x < p.width; ++x) {
            const size_t i = size_t(y)*p.width+x;
            const int tile = y/32*tx+x/64;
            alive += now[i] == 1; nonzero += now[i] != 0;
            nz[tile] += now[i] != 0;
            if (old) { prev[tile] += (*old)[i] != 0; diff[tile] += (*old)[i] != now[i]; }
        }
    // Scatter occupied-tile influence, independently of the GPU gather reduction.
    for (int tile = 0; tile < count; ++tile) if (nz[tile]) {
        auto xs = tile_geometry::influence(tile%tx,64,p.width,p.rule.radius,p.boundary==Boundary::Torus);
        auto ys = tile_geometry::influence(tile/tx,32,p.height,p.rule.radius,p.boundary==Boundary::Torus);
        for (int y = 0; y < ys.count; ++y)
            for (int x = 0; x < xs.count; ++x) influence[ys.ids[y]*tx+xs.ids[x]] = 1;
    }
    int nt = 0, ct = 0, it = 0, work = 0;
    for (int tile = 0; tile < count; ++tile) {
        changed += diff[tile]; nt += nz[tile] != 0; ct += diff[tile] != 0;
        it += influence[tile]; work += influence[tile] || prev[tile];
        ++bins[!nz[tile] ? 0 : nz[tile] <= 16 ? 1 : nz[tile] <= 128 ? 2 : nz[tile] <= 512 ? 3 : 4];
    }
    std::fprintf(stderr,
        "[EVOLVE_EXPECT] generation=%d alive=%llu nonzero=%llu decay=%llu changed=%llu "
        "nonzero_tiles=%d changed_tiles=%d influence_tiles=%d work_tiles=%d "
        "empty_tiles=%llu tiles_1_16=%llu tiles_17_128=%llu tiles_129_512=%llu tiles_513_up=%llu fixed=%d\n",
        generation,alive,nonzero,nonzero-alive,changed,nt,ct,it,work,
        bins[0],bins[1],bins[2],bins[3],bins[4],int(generation>0 && changed==0));
}

int main() {
    check(cudaFree(nullptr));
    int id = 0;
    for (Boundary boundary : {Boundary::Dead, Boundary::Torus})
    for (int pattern = 0; pattern < 10; ++pattern) {
        Rule life{1,2,false,2,3,3,3,Shape::Moore};
        Problem p{pattern==3 || pattern==8 ? 321 : 129, pattern==3 || pattern==8 ? 161 : 65,
                  life,boundary,pattern==0 ? 0 : pattern==4 ? 17 : 33};
        if (pattern==4) p.rule.states=16;
        if (pattern==5) p.rule=Rule{1,4,false,3,5,2,2,Shape::Moore};
        if (pattern==6 || pattern==7) p.rule=Rule{5,4,true,12,34,3,9,pattern==6 ? Shape::Moore : Shape::VonNeumann};
        if (pattern==8) p.rule=Rule{1,2,false,0,8,1,8,Shape::Moore};
        std::vector<uint8_t> initial(size_t(p.width)*p.height,0);
        auto put=[&](int x,int y,int value) { initial[size_t(y%p.height)*p.width+x%p.width]=uint8_t(value); };
        if (pattern==1) { put(62,30,1);put(63,30,1);put(62,31,1);put(63,31,1); }
        if (pattern==2) { put(62,31,1);put(63,31,1);put(64,31,1); }
        if (pattern==3) { int x=p.width-2,y=p.height-2;put(x+1,y,1);put(x+2,y+1,1);put(x,y+2,1);put(x+1,y+2,1);put(x+2,y+2,1); }
        if (pattern==4) { put(0,0,2);put(63,31,15);put(128,64,7); }
        if (pattern>=5 && pattern<=7)
            for (size_t i=0;i<initial.size();++i) initial[i]=uint8_t((i*17+i/7)%p.rule.states);
        if (pattern==8) for (int x=0;x<p.width;++x) put(x,31,1);
        std::fprintf(stderr,"[EVOLVE_TEST] id=%d steps=%d\n",id++,p.steps);
        auto reference=initial;
        expected(0,p,reference,nullptr);
        Problem one=p; one.steps=1;
        for(int t=1;t<=p.steps;++t) {
            auto previous=reference;
            run_reference(reference.data(),one);
            expected(t,p,reference,&previous);
        }
        uint8_t *actual=nullptr;
        check(cudaMallocHost(&actual,initial.size()));
        std::copy(initial.begin(),initial.end(),actual);
        run(actual,p);
        check(cudaDeviceSynchronize());
        if (!std::equal(reference.begin(),reference.end(),actual)) {
            std::fprintf(stderr,"FAIL final grid id=%d\n",id-1); return 1;
        }
        check(cudaFreeHost(actual));
    }
    std::fprintf(stderr,"PASS %d/%d evolution final-grid checks\n",id,id);
}
