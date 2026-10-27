#pragma once
// Included inside the implementation namespace. Dense R1 C=2 only.
// Source and destination are distinct allocations, reused across byte/bit formats.
__global__ void pack_r1_bits(const uint8_t *src, uint32_t *dst, int width, int height) {
    const int lane = int(threadIdx.x) & 31;
    const int pitch = (width+31)/32;
    // Bounded grid, several words per warp: avoid millions of tiny conversion blocks.
    for (int word = int(blockIdx.x)*(int(blockDim.x)/32)+int(threadIdx.x)/32;
         word < pitch*height; word += int(gridDim.x)*(int(blockDim.x)/32)) {
        const int y = word/pitch, x = (word%pitch)*32+lane;
        const bool live = x < width && src[size_t(y)*width+x] == 1;
        const uint32_t bits = __ballot_sync(0xffffffffu,live);
        if (lane == 0) dst[word] = bits;
    }
}
__global__ void unpack_r1_bits(const uint32_t *src, uint8_t *dst, int width, int height) {
    const int lane = int(threadIdx.x)&31, pitch = (width+31)/32;
    for (int word = int(blockIdx.x)*(int(blockDim.x)/32)+int(threadIdx.x)/32;
         word < pitch*height; word += int(gridDim.x)*(int(blockDim.x)/32)) {
        const int y = word/pitch, x = (word%pitch)*32+lane;
        if (x < width) dst[size_t(y)*width+x] = uint8_t((src[word]>>lane)&1u);
    }
}

template <bool Torus>
__device__ __forceinline__ uint32_t bit_word(const uint32_t *src, int wx, int y,
                                           int pitch, int height) {
    if (wx < 0 || wx >= pitch) return 0;
    if constexpr (Torus) { if (y < 0) y += height; else if (y >= height) y %= height; }
    else if (y < 0 || y >= height) return 0;
    return src[size_t(y)*pitch+wx];
}
struct BitRow { uint32_t left, centre, right; };
template <bool Torus>
__device__ __forceinline__ BitRow bit_row(const uint32_t *src, int wx, int y,
                                         int width, int height) {
    const int pitch = (width+31)/32, lane = int(threadIdx.x)&31;
    const uint32_t c = bit_word<Torus>(src,wx,y,pitch,height);
    // Every lane participates, including padding lanes. Only warp-edge lanes
    // load the extra word; no block-wide shared-memory exchange is needed.
    uint32_t prev = __shfl_up_sync(0xffffffffu,c,1);
    uint32_t next = __shfl_down_sync(0xffffffffu,c,1);
    if (lane == 0) prev = bit_word<Torus>(src,wx-1,y,pitch,height);
    if (lane == 31) next = bit_word<Torus>(src,wx+1,y,pitch,height);
    uint32_t left = (c<<1) | (prev>>31), right = (c>>1) | (next<<31);
    if constexpr (Torus) {
        // Partial last words wrap at the REAL width, not the padded pitch.
        if (wx == 0)
            left = (c<<1) | ((bit_word<true>(src,pitch-1,y,pitch,height) >> ((width-1)&31))&1u);
        if (wx == pitch-1)
            right = (c>>1) | ((bit_word<true>(src,0,y,pitch,height)&1u) << ((width-1)&31));
    }
    return {left,c,right};
}
template <int Rows, bool Moore, bool Torus>
__global__ void step_r1_bits(const uint32_t *src, uint32_t *dst, Problem p) {
    static_assert(Rows == 1 || Rows == 4, "bounded spatial reuse variants");
    const int wx = int(blockIdx.x)*32 + int(threadIdx.x);
    const int y0 = (int(blockIdx.y)*int(blockDim.y)+int(threadIdx.y))*Rows;
    const int pitch = (p.width+31)/32;
    // Rolling register window within ONE generation, not temporal blocking.
    BitRow up = bit_row<Torus>(src,wx,y0-1,p.width,p.height);
    BitRow mid = bit_row<Torus>(src,wx,y0,p.width,p.height);
#pragma unroll
    for (int row=0; row<Rows; ++row) {
        const int y = y0+row;
        BitRow down = bit_row<Torus>(src,wx,y+1,p.width,p.height);
        r1_bits::Count n;
        if constexpr (Moore)
            n = r1_bits::count9(up.left,up.centre,up.right,mid.left,mid.right,
                               down.left,down.centre,down.right,p.rule.count_centre ? mid.centre : 0);
        else
            n = r1_bits::count9(0,up.centre,0,mid.left,mid.right,0,down.centre,0,
                               p.rule.count_centre ? mid.centre : 0);
        const uint32_t out = r1_bits::next(mid.centre,n,p.rule.survive_lo,p.rule.survive_hi,
                                         p.rule.birth_lo,p.rule.birth_hi);
        if (wx < pitch && y < p.height)
            dst[size_t(y)*pitch+wx] = out & r1_bits::mask(p.width,wx);
        up = mid; mid = down;
    }
}

struct BitplaneExperiment {
    bool requested = false, selected = false, timed = false;
    int rows = 4;
    const char *reason = "disabled";
    cudaEvent_t events[4]{};
    float pack_ms=0, evolve_ms=0, unpack_ms=0;
    BitplaneExperiment() {
        const char *mode = std::getenv("PP_BITPLANE");
        requested = mode && std::strcmp(mode,"1")==0;
        if (mode && !requested && std::strcmp(mode,"0")!=0) {
            std::fprintf(stderr,"hw3-1: PP_BITPLANE must be 0 or 1\n"); std::exit(EXIT_FAILURE);
        }
        const char *r = std::getenv("PP_BIT_ROWS");
        if (r && std::strcmp(r,"1")==0) rows=1;
        else if (r && std::strcmp(r,"4")!=0) {
            std::fprintf(stderr,"hw3-1: PP_BIT_ROWS must be 1 or 4\n"); std::exit(EXIT_FAILURE);
        }
    }
    void prepare(bool profile) {
        timed = selected && profile;
        if (timed) for (auto &e:events) check(cudaEventCreate(&e),"create bit-plane event");
    }
    void mark(int i) { if (timed) check(cudaEventRecord(events[i],0),"record bit-plane event"); }
    void finish() {
        if (!timed) return;
        check(cudaEventSynchronize(events[3]),"wait bit-plane events");
        check(cudaEventElapsedTime(&pack_ms,events[0],events[1]),"time bit pack");
        check(cudaEventElapsedTime(&evolve_ms,events[1],events[2]),"time bit evolution");
        check(cudaEventElapsedTime(&unpack_ms,events[2],events[3]),"time bit unpack");
        for (auto e:events) check(cudaEventDestroy(e),"destroy bit-plane event");
    }
    void report(bool profile) const {
        if (profile) std::fprintf(stderr,
            "[PP_BITPLANE] requested=%d selected=%d rows=%d reason=%s pack_ms=%.6f evolve_ms=%.6f unpack_ms=%.6f\n",
            int(requested),int(selected),rows,reason,pack_ms,evolve_ms,unpack_ms);
    }
};
