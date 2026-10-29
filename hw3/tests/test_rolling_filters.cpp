// CPU-only validation of rolling arithmetic and producer/output coverage.
// The oracle directly enumerates the mathematical neighbourhood, independently
// of the two diagonal boxes and packed vertical/word-prefix representation.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "../ca.hpp"
#include "../r1_packed.hpp"
#include "../rolling_filters.hpp"

static void require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static uint8_t cell(const std::vector<uint8_t> &grid, const Problem &p, int x, int y) {
    if (p.boundary == Boundary::Torus) {
        x = (x % p.width + p.width) % p.width;
        y = (y % p.height + p.height) % p.height;
    }
    return x < 0 || x >= p.width || y < 0 || y >= p.height ? 0 : grid[y*p.width+x];
}
static int direct(const std::vector<uint8_t> &grid, const Problem &p, int x, int y) {
    int count = 0, r = p.rule.radius;
    for (int dy=-r; dy<=r; ++dy) {
        const int span = p.rule.shape == Shape::Moore ? r : r-std::abs(dy);
        for (int dx=-span; dx<=span; ++dx)
            if (dx || dy || p.rule.count_centre) count += cell(grid,p,x+dx,y+dy)==1;
    }
    return count;
}

template <int Width, int Height>
static std::vector<int> emulate(const std::vector<uint8_t> &grid, const Problem &p, int x0, int y0) {
    const int r=p.rule.radius, left=rolling_filters::halo_x(r);
    const int pitch=rolling_filters::source_pitch<Width,Height>(r), wp=pitch/4;
    std::vector<uint32_t> source(wp*(Height+2*r));
    for (int y=0; y<Height+2*r; ++y)
        for (int x=0; x<pitch; ++x)
            source[y*wp+x/4] |= uint32_t(cell(grid,p,x0+x-left,y0+y-r)) << (8*(x&3));
    auto raw = [&](int x,int y) -> uint8_t {
        require(x>=0 && x<pitch && y>=0 && y<Height+2*r,"raw bounds");
        return uint8_t(source[y*wp+x/4] >> (8*(x&3)));
    };
    std::vector<int> result(Width*Height,-1), visits(Width*Height);
    if (p.rule.shape == Shape::Moore) {
        std::vector<uint32_t> vertical(wp*Height);
        for (int first=0; first<Height; first+=rolling_filters::line_rows)
            for (int word=0; word<wp; ++word) {
                uint32_t sum=0;
                for (int i=0;i<=2*r;++i) sum+=r1_packed::equal(source[(first+i)*wp+word],1);
                const int end=std::min(Height,first+rolling_filters::line_rows);
                for(int y=first;y<end;++y) {
                    vertical[y*wp+word]=sum;
                    if(y+1<end) sum=rolling_filters::slide_bytes(sum,
                        r1_packed::equal(source[y*wp+word],1),
                        r1_packed::equal(source[(y+2*r+1)*wp+word],1));
                }
            }
        for(int y=0;y<Height;++y) {
            std::vector<unsigned> q(wp+1);
            for(int word=0;word<wp;++word) q[word+1]=q[word]+rolling_filters::byte_prefix(vertical[y*wp+word],4);
            auto endpoint=[&](int e) {
                require(e>=0 && e<=pitch,"Moore endpoint bounds");
                return q[e/4]+((e&3)?rolling_filters::byte_prefix(vertical[y*wp+e/4],e&3):0u);
            };
            for(int x=0;x<Width;++x) {
                result[y*Width+x]=int(endpoint(left+x+r+1))-int(endpoint(left+x-r));
                ++visits[y*Width+x];
            }
        }
    } else {
        const int iw=Width+r, ih=Height+r, up=rolling_filters::align2(iw);
        std::vector<uint16_t> u(up*ih);
        std::vector<int> writes(up*ih);
        auto at=[&](int x,int y)->uint16_t {
            require(x>=0 && x<iw && y>=0 && y<ih,"VN intermediate bounds");
            require(writes[y*up+x]==1,"VN intermediate producer coverage");
            return u[y*up+x];
        };
        for(int d=0;d<iw+ih-1;++d) {
            int x=d<iw?d:0, y=d<iw?0:d-iw+1, sum=0;
            for(int i=0;i<=r;++i) sum+=raw(left-r+x+i,y+i)==1;
            while(x<iw && y<ih) {
                u[y*up+x]=rolling_filters::pack_diagonal(unsigned(sum),raw(left+x,y+r)==1);
                ++writes[y*up+x];
                if(x+1>=iw || y+1>=ih) break;
                sum-=raw(left-r+x,y)==1;
                sum+=raw(left+x+1,y+r+1)==1;
                ++x; ++y;
            }
        }
        constexpr int rows=rolling_filters::line_rows;
        constexpr int starts=(Width+rows-1+31)&~31;
        for(int task=0;task<starts*((Height+rows-1)/rows);++task) {
            const int sx=task%starts-(rows-1), top=(task/starts)*rows;
            const int bottom=std::min(top+rows,Height)-1, count=bottom-top+1;
            const int first=std::max(0,-sx), end=std::min(Width-sx,count);
            if(first>=end) continue;
            int x=sx+first,y=bottom-first,a=0,b=0;
            for(int j=0;j<=r;++j) a+=rolling_filters::diagonal_plus(at(x+j,y+r-j));
            for(int j=0;j<r;++j) b+=rolling_filters::diagonal_minus(at(x+1+j,y+r-j));
            for(int step=first;step<end;++step) {
                result[y*Width+x]=a+b;
                ++visits[y*Width+x];
                if(step+1<end) {
                    a-=rolling_filters::diagonal_plus(at(x,y+r));
                    a+=rolling_filters::diagonal_plus(at(x+r+1,y-1));
                    b-=rolling_filters::diagonal_minus(at(x+1,y+r));
                    b+=rolling_filters::diagonal_minus(at(x+r+1,y));
                    ++x; --y;
                }
            }
        }
    }
    for(int y=0;y<Height;++y) for(int x=0;x<Width;++x) {
        require(visits[y*Width+x]==1,"output coverage exactly once");
        result[y*Width+x]-=int(!p.rule.count_centre && raw(left+x,y+r)==1);
    }
    require(rolling_filters::shared_bytes<Width,Height>(r,p.rule.shape==Shape::Moore)
        >=source.size()*sizeof(uint32_t),"scratch geometry");
    return result;
}

template <int Width,int Height>
static unsigned exercise(std::mt19937 &rng) {
    unsigned checks=0;
    for(int r=1;r<=32;++r) for(int boundary=0;boundary<2;++boundary)
        for(int shape=0;shape<2;++shape) for(int pattern=0;pattern<3;++pattern) {
            Problem p{};
            // Include small grids whose padded tile wraps repeatedly, odd row
            // pitches, and partial tiles, including narrow one-cell tails.
            p.width=2*r+1+(r%7); p.height=2*r+1+((r*3)%11);
            if(pattern==2) { p.width+=Width; p.height+=Height; }
            p.rule.radius=r; p.rule.states=2+(r%15); p.rule.count_centre=r&1;
            p.rule.shape=shape?Shape::VonNeumann:Shape::Moore;
            p.boundary=boundary?Boundary::Torus:Boundary::Dead;
            std::vector<uint8_t> grid(p.width*p.height);
            for(auto &v:grid) v=pattern==0?0:pattern==1?1:uint8_t(rng()%p.rule.states);
            const int x0=pattern==2?Width:0, y0=pattern==2?Height:0;
            const auto got=emulate<Width,Height>(grid,p,x0,y0);
            // Verify edges, radius boundaries, and random interior/tail cells
            // with the independent direct neighbourhood definition.
            for(int sample=0;sample<40;++sample) {
                const int valid_w=std::min(Width,p.width-x0), valid_h=std::min(Height,p.height-y0);
                const int x=sample<4?((sample&1)?valid_w-1:0):int(rng()%valid_w);
                const int y=sample<4?((sample&2)?valid_h-1:0):int(rng()%valid_h);
                const int want=direct(grid,p,x0+x,y0+y);
                if(got[y*Width+x]!=want) {
                    std::fprintf(stderr,"R%d shape%d boundary%d pattern%d tile%dx%d xy%d,%d expected%d got%d\n",
                        r,shape,boundary,pattern,Width,Height,x,y,want,got[y*Width+x]);
                    std::exit(1);
                }
                ++checks;
            }
        }
    return checks;
}
int main() {
    std::mt19937 rng(20261029);
    const unsigned checks=exercise<64,32>(rng)+exercise<128,64>(rng);
    std::printf("PASS %u direct neighbourhood checks; R1..32, both shapes/boundaries, packed and diagonal rolling, source/intermediate/output bounds and coverage (CPU only)\n",checks);
}
