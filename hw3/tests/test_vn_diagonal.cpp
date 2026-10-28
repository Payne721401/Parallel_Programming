// CPU-only: validate the exact eight-endpoint query against direct diamonds.
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>
#include "../vn_diagonal.hpp"
int main() {
    std::mt19937 rng(20261028);
    unsigned checks = 0;
    for (int r = 1; r <= 32; ++r) {
        const int rw=64+2*r, rh=32+2*r, pitch=rw+1;
        for (int pattern=0; pattern<3; ++pattern) {
            std::vector<int> cells(rw*rh);
            for (auto &v:cells) v=pattern==0 ? 0 : pattern==1 ? 1 : int(rng()%3==0);
            std::vector<uint16_t> q((rh+1)*pitch), a(q.size()), b(q.size());
            for (int y=0;y<rh;++y)
                for (int x=0;x<rw;++x) q[(y+1)*pitch+x+1]=q[(y+1)*pitch+x]+cells[y*rw+x];
            // Independent row-order construction, unlike the GPU diagonal scans.
            for (int y=1;y<=rh;++y)
                for (int x=0;x<=rw;++x) {
                    a[y*pitch+x]=q[y*pitch+x]+(x>0?a[(y-1)*pitch+x-1]:0);
                    b[y*pitch+x]=q[y*pitch+x]+(x<rw?b[(y-1)*pitch+x+1]:0);
                }
            // Check producer coverage and in-place B construction independently
            // against the row-order oracle above; this does not emulate CUDA sync.
            auto inplace=q;
            std::vector<uint16_t> scanned(q.size());
            for (int pass=0;pass<2;++pass) {
                auto &out=pass==0?scanned:inplace;
                std::vector<int> visits(q.size());
                for (int d=0;d<pitch+rh-1;++d) {
                    int row=d<pitch?1:d-pitch+2;
                    int col=d<pitch?d:(pass==0?0:rw);
                    unsigned carry=0;
                    while (row<=rh && col>=0 && col<pitch) {
                        const int index=row*pitch+col;
                        carry+=inplace[index];
                        out[index]=uint16_t(carry);
                        ++visits[index];
                        ++row; col+=pass==0?1:-1;
                    }
                }
                const auto &oracle=pass==0?a:b;
                for (int y=1;y<=rh;++y)
                    for (int x=0;x<pitch;++x)
                        if (visits[y*pitch+x]!=1 || out[y*pitch+x]!=oracle[y*pitch+x]) return 2;
            }
            for (int y=0;y<32;++y)
                for (int x=0;x<64;++x) {
                    int expected=0;
                    for (int dy=-r;dy<=r;++dy) {
                        const int span=r-std::abs(dy);
                        for (int dx=-span;dx<=span;++dx) expected+=cells[(y+r+dy)*rw+x+r+dx];
                    }
                    const int actual=vn_diagonal::count(a.data(),b.data(),pitch,x,y,r);
                    if (actual!=expected) {
                        std::fprintf(stderr,"R=%d x=%d y=%d expected=%d actual=%d\n",r,x,y,expected,actual);
                        return 1;
                    }
                    ++checks;
                }
        }
    }
    std::printf("PASS %u diamond queries (R1..32, zero/one/random; CPU only)\n",checks);
}
