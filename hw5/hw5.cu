// HW 5: YOUR FILE. Everything here is yours to change, behind the one function main() calls:
// run(). mixtral.hpp says what run() is given and what it must leave behind.
//
// A ROW is one token at one position of one prompt. forward() runs one batch of rows through the
// model: the prefill is every prompt's tokens, CHUNK rows at a time, and each decode step is one
// row per prompt still generating.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#ifdef HW5_NSIGHT
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mixtral.hpp"

namespace {

constexpr int CHUNK = 4096;   // the most rows in one forward pass

void check(cudaError_t e, const char *what)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "hw5.cu: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <class T>
T *dalloc(size_t n)
{
    T *p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return p;
}

int blocks(size_t n, int per) { return int((n + per - 1) / per); }

// ---- kernels ------------------------------------------------------------------------------------

// x[r] = embed[token[r]]. Grid: rows.
__global__ void embed_kernel(const int *token, const __nv_bfloat16 *embed, float *x)
{
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) x[size_t(r) * D + i] = __bfloat162float(embed[size_t(token[r]) * D + i]);
}

// y[r] = RMSNorm(x[r]; scale). Grid: rows, block: 256.
__global__ void rmsnorm_kernel(const float *x, const __nv_bfloat16 *scale, float *y)
{
    __shared__ float sum[256];
    const int r = blockIdx.x, t = threadIdx.x;
    const float *xr = x + size_t(r) * D;
    float s = 0.f;
    for (int i = t; i < D; i += 256) s += xr[i] * xr[i];
    sum[t] = s;
    __syncthreads();
    for (int half = 128; half > 0; half /= 2) {
        if (t < half) sum[t] += sum[t + half];
        __syncthreads();
    }
    const float inv = 1.f / sqrtf(sum[0] / D + RMS_EPS);
    for (int i = t; i < D; i += 256) y[size_t(r) * D + i] = xr[i] * inv * __bfloat162float(scale[i]);
}

// y[M][N] = x[M][K] * W[N][K]^T, or y += it. A block of 256 threads computes a 64x64 tile of y,
// each thread a 4x4 piece of it, from 64x32 slices of x and W staged in shared memory.
constexpr int TM = 64, TN = 64, TK = 32;
__global__ void linear_kernel(const float *x, const __nv_bfloat16 *W, float *y, int M, int N, int K, bool add)
{
    __shared__ float xs[TM][TK + 1], ws[TN][TK + 1];
    const int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN;
    const int tm = threadIdx.x / 16 * 4, tn = threadIdx.x % 16 * 4;   // this thread's 4x4 piece
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = threadIdx.x; i < TM * TK; i += 256) {
            const int r = i / TK, k = i % TK;
            xs[r][k] = m0 + r < M ? x[size_t(m0 + r) * K + k0 + k] : 0.f;
            ws[r][k] = n0 + r < N ? __bfloat162float(W[size_t(n0 + r) * K + k0 + k]) : 0.f;
        }
        __syncthreads();
        for (int k = 0; k < TK; k++)
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] += xs[tm + i][k] * ws[tn + j][k];
        __syncthreads();
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            const int m = m0 + tm + i, n = n0 + tn + j;
            if (m < M && n < N) y[size_t(m) * N + n] = add ? y[size_t(m) * N + n] + acc[i][j] : acc[i][j];
        }
}

// Many rows (prefill): y[M][N] = x[M][K] * W[N][K]^T on the FP32 cores, structured as the 2D
// block-tiled, vectorized SGEMM of Simon Boehm's SGEMM_CUDA (kernels 6 and 9: 128 x 128 block
// tile, BK = 16, 256 threads each holding an 8 x 8 register tile). Nsight Compute showed the 64 x 64
// kernel above at 19 TFLOPS with the FMA pipe 46% busy and the LSU 75%: each thread did 16 FMAs per
// 8 scalar shared loads. Here a thread does 64 FMAs per four float4 shared loads. Both operands
// are K-contiguous, so both tiles are stored transposed (k-major) in shared memory, as Boehm does for
// A; W's 8 bf16 per 16-byte load are widened exactly by bit shifts. Each output is one fmaf chain
// over k in increasing order, the same order as linear_kernel.
//
// Nsight Compute on this kernel (case 03, M = 4096) then showed 28.6 TFLOPS with 87% of the
// instructions FMAs but only 62% of the issue slots used: occupancy 25% (127 registers), warps
// waiting on the next tile's global loads (long scoreboard) and on shared memory (short scoreboard,
// L1/TEX 86% busy). Three changes, none of which moves a single FMA of an output's chain:
//   - the next tile's global loads go into registers before the current tile is computed and are
//     stored to shared memory after it (the global-to-register double buffering of SGEMM_CUDA
//     kernels 11/12 and CUTLASS's SIMT main loop), so their latency hides behind 16 x 64 FMAs;
//   - a thread's 8 x 8 tile is two 4 x 4 halves 64 apart in each dimension (the CUTLASS SIMT and
//     SGEMM_CUDA warp-tile layout), so the 16 threads of a half-warp read 16 consecutive float4 of a
//     row of ws, which is conflict-free, where 8 consecutive columns per thread were 2-way;
//   - the shared rows are padded by 4 floats, which halves the 4-way bank conflict of the
//     transposed stores and keeps every float4 aligned;
// and at most 128 registers, so two blocks keep fitting on an SM with the prefetch registers added.
constexpr int GBM = 128, GBN = 128, GBK = 16, GT = 8, GH = GT / 2, GTHREADS = (GBM / GT) * (GBN / GT), GPAD = 4;
static_assert(GTHREADS == 256 && GBM * GBK / 4 == 2 * GTHREADS && GBN * GBK / 8 == GTHREADS, "tile loads per thread");
static_assert(GBM / 2 == (GBM / GT) * GH && GBN / 2 == (GBN / GT) * GH, "two halves of 16 x 4 rows and columns");

// bf16 is the top half of an fp32: widening a packed pair is exact, two bit operations.
__device__ __forceinline__ float bf16_lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16_hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__global__ void __launch_bounds__(GTHREADS, 2) sgemm_kernel(const float *x, const __nv_bfloat16 *W, float *y,
                                                           int M, int N, int K, bool add)
{
    __shared__ __align__(16) float xs[GBK][GBM + GPAD];
    __shared__ __align__(16) float ws[GBK][GBN + GPAD];
    const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN, t = threadIdx.x;
    // This thread's rows tr*4 + i and GBM/2 + tr*4 + i, columns tc*4 + j and GBN/2 + tc*4 + j.
    const int tr = t / (GBN / GT), tc = t % (GBN / GT);
    // The tile loads: x rows xr0, xr0 + 64 at column xc; W row wr at column wc.
    const int xr0 = t / (GBK / 4), xc = t % (GBK / 4) * 4, wr = t / (GBK / 8), wc = t % (GBK / 8) * 8;
    const float *xp0 = x + size_t(m0 + xr0) * K + xc, *xp1 = x + size_t(m0 + xr0 + GBM / 2) * K + xc;
    const bool xv0 = m0 + xr0 < M, xv1 = m0 + xr0 + GBM / 2 < M;
    const __nv_bfloat16 *wp = W + size_t(n0 + wr) * K + wc;
    float4 xa, xb;
    uint4 wq;
    auto fetch = [&](int k0) {
        xa = xv0 ? *reinterpret_cast<const float4 *>(xp0 + k0) : make_float4(0.f, 0.f, 0.f, 0.f);
        xb = xv1 ? *reinterpret_cast<const float4 *>(xp1 + k0) : make_float4(0.f, 0.f, 0.f, 0.f);
        wq = __ldg(reinterpret_cast<const uint4 *>(wp + k0));
    };
    auto stash = [&]() {
        xs[xc][xr0] = xa.x; xs[xc + 1][xr0] = xa.y; xs[xc + 2][xr0] = xa.z; xs[xc + 3][xr0] = xa.w;
        xs[xc][xr0 + GBM / 2] = xb.x; xs[xc + 1][xr0 + GBM / 2] = xb.y;
        xs[xc + 2][xr0 + GBM / 2] = xb.z; xs[xc + 3][xr0 + GBM / 2] = xb.w;
        ws[wc][wr] = bf16_lo(wq.x); ws[wc + 1][wr] = bf16_hi(wq.x); ws[wc + 2][wr] = bf16_lo(wq.y); ws[wc + 3][wr] = bf16_hi(wq.y);
        ws[wc + 4][wr] = bf16_lo(wq.z); ws[wc + 5][wr] = bf16_hi(wq.z); ws[wc + 6][wr] = bf16_lo(wq.w); ws[wc + 7][wr] = bf16_hi(wq.w);
    };
    float acc[GT][GT] = {};
    fetch(0);
    stash();
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += GBK) {
        const bool more = k0 + GBK < K;
        if (more) fetch(k0 + GBK);   // in flight while this tile is computed
        #pragma unroll
        for (int k = 0; k < GBK; k++) {
            const float4 a0 = *reinterpret_cast<const float4 *>(&xs[k][tr * GH]);
            const float4 a1 = *reinterpret_cast<const float4 *>(&xs[k][GBM / 2 + tr * GH]);
            const float4 b0 = *reinterpret_cast<const float4 *>(&ws[k][tc * GH]);
            const float4 b1 = *reinterpret_cast<const float4 *>(&ws[k][GBN / 2 + tc * GH]);
            const float a[GT] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[GT] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < GT; i++)
                #pragma unroll
                for (int j = 0; j < GT; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();   // every thread is done reading this tile
        if (more) {
            stash();
            __syncthreads();
        }
    }
    #pragma unroll
    for (int i = 0; i < GT; i++) {
        const int m = m0 + (i < GH ? tr * GH + i : GBM / 2 + tr * GH + i - GH);
        if (m >= M) continue;
        float *row = y + size_t(m) * N + n0;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            float4 *dst = reinterpret_cast<float4 *>(row + h * (GBN / 2) + tc * GH);
            float4 v = make_float4(acc[i][h * GH], acc[i][h * GH + 1], acc[i][h * GH + 2], acc[i][h * GH + 3]);
            if (add) {
                const float4 o = *dst;
                v = make_float4(o.x + v.x, o.y + v.y, o.z + v.z, o.w + v.w);
            }
            *dst = v;
        }
    }
}

// Few rows (decode): a projection reads all of W for little arithmetic. The tiling follows
// TensorRT-LLM's weightOnlyBatchedGemv (v0.7.1 kernel.h), whose threads each hold an accumulator
// of Batch x NPerBlock: every 16-byte activation load is used against NPerBlock weight columns,
// and every weight load against all Batch rows. Nsight Compute showed the one-column version
// spending its time on L1 (82-98% busy, 65x more L1 load traffic than weight bytes) re-reading x,
// with DRAM at 4-43%, so x is now loaded once for C columns.
//
// One warp computes C consecutive output columns for up to R rows. Each lane loads 8 bf16 of each
// column's row of W at once (16 bytes), widens them exactly by moving the bits into an fp32's top
// half (bf16 is fp32's top half), and applies them to the same 8 values of every row's x (two
// float4 loads, shared by the C columns). Every product and sum is fp32. Each lane sums its 8-wide
// chunks in increasing k and the warp adds the 32 lane sums in one fixed tree, the same for every
// R and C, so an output does not depend on how many rows share the launch.
constexpr int SKINNY_WARPS = 8, SKINNY_MAX_ROWS = 64;
static_assert(D % 256 == 0 && Q_DIM % 256 == 0 && I % 256 == 0, "every K is a whole number of 32 lanes x 8");

// At least 2 blocks per SM: ptxas gave <8,4> 148 and <16,2> 144 registers, which Nsight Compute
// showed holding the SM to one block, 12.5% of its warps, with DRAM at 9-19%.
template <int R, int C>
__global__ void __launch_bounds__(SKINNY_WARPS * 32, 2) skinny_kernel(const float *x, const __nv_bfloat16 *W, float *y,
                                                                   int M, int N, int K, bool add)
{
    const int out0 = (blockIdx.x * SKINNY_WARPS + threadIdx.x / 32) * C;
    if (out0 >= N) return;   // uniform within each warp; N is a multiple of C
    const int lane = threadIdx.x % 32, first = blockIdx.y * R, k4 = K / 4;
    const uint4 *w = reinterpret_cast<const uint4 *>(W + size_t(out0) * K);
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    int row_off[R];
    #pragma unroll
    for (int r = 0; r < R; r++) row_off[r] = min(first + r, M - 1) * k4;
    float sum[R][C];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < C; j++) sum[r][j] = 0.f;
    for (int c = lane; c < K / 8; c += 32) {
        float wf[C][8];
        #pragma unroll
        for (int j = 0; j < C; j++) {
            const uint4 p = __ldg(w + size_t(j) * (K / 8) + c);
            wf[j][0] = bf16_lo(p.x); wf[j][1] = bf16_hi(p.x); wf[j][2] = bf16_lo(p.y); wf[j][3] = bf16_hi(p.y);
            wf[j][4] = bf16_lo(p.z); wf[j][5] = bf16_hi(p.z); wf[j][6] = bf16_lo(p.w); wf[j][7] = bf16_hi(p.w);
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const float4 a = __ldg(x4 + row_off[r] + 2 * c), b = __ldg(x4 + row_off[r] + 2 * c + 1);
            #pragma unroll
            for (int j = 0; j < C; j++) {
                float s = sum[r][j];
                s = fmaf(a.x, wf[j][0], s); s = fmaf(a.y, wf[j][1], s); s = fmaf(a.z, wf[j][2], s); s = fmaf(a.w, wf[j][3], s);
                s = fmaf(b.x, wf[j][4], s); s = fmaf(b.y, wf[j][5], s); s = fmaf(b.z, wf[j][6], s); s = fmaf(b.w, wf[j][7], s);
                sum[r][j] = s;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < C; j++) {
            float s = sum[r][j];
            for (int d = 16; d; d /= 2) s += __shfl_down_sync(0xffffffffu, s, d);
            if (lane == 0 && first + r < M) {
                float *dst = y + size_t(first + r) * N + out0 + j;
                *dst = add ? *dst + s : s;
            }
        }
}

template <int R, int C>
void skinny(const float *x, const __nv_bfloat16 *W, float *y, int M, int N, int K, bool add, cudaStream_t stream)
{
    skinny_kernel<R, C><<<dim3(blocks(N, SKINNY_WARPS * C), blocks(M, R)), SKINNY_WARPS * 32, 0, stream>>>(x, W, y, M, N, K, add);
}

// Every K here (D, Q_DIM, I) is a multiple of TK and of 256, and every N (QKV_DIM, D, 2I, VOCAB)
// of 4. x, W and y are the starts of allocations or of whole rows in them, so the 16-byte loads
// of skinny_kernel are aligned. Up to 16 rows share one pass over W; 17-64 rows take 2-4 passes,
// which keeps the accumulators (R x C = 32) and so the occupancy where Nsight showed R = 32 lose it.
static_assert(QKV_DIM % 4 == 0 && D % 4 == 0 && (2 * I) % 4 == 0 && VOCAB % 4 == 0, "C divides every N");
void linear(const float *x, const __nv_bfloat16 *W, float *y, int M, int N, int K, bool add = false, cudaStream_t stream = nullptr)
{
    if (M <= SKINNY_MAX_ROWS) {
        // One row: x is a single vector, so reuse buys little and C = 1 keeps 4x the warps
        // (case 12 ran 6% slower with C = 4).
        if (M == 1) skinny<1, 1>(x, W, y, M, N, K, add, stream);
        else if (M <= 2) skinny<2, 4>(x, W, y, M, N, K, add, stream);
        else if (M <= 4) skinny<4, 4>(x, W, y, M, N, K, add, stream);
        else if (M <= 8) skinny<8, 4>(x, W, y, M, N, K, add, stream);
        else skinny<16, 2>(x, W, y, M, N, K, add, stream);
        return;
    }
    if (N % GBN == 0 && K % GBK == 0) {   // every N and K of the model
        sgemm_kernel<<<dim3(N / GBN, blocks(M, GBM)), GTHREADS, 0, stream>>>(x, W, y, M, N, K, add);
        return;
    }
    dim3 grid(blocks(N, TN), blocks(M, TM));
    linear_kernel<<<grid, 256, 0, stream>>>(x, W, y, M, N, K, add);
}

// Rotates q's 32 heads and k's 8 by each row's position. Grid: (rows, N_HEAD + N_KV), block: 64.
__global__ void rope_kernel(float *qkv, const int *pos, const float *rcos, const float *rsin)
{
    const int r = blockIdx.x, hd = blockIdx.y, i = threadIdx.x;
    float *v = qkv + size_t(r) * QKV_DIM + hd * HEAD_DIM;   // the K heads follow the Q heads
    const float c = rcos[pos[r] * (HEAD_DIM / 2) + i], s = rsin[pos[r] * (HEAD_DIM / 2) + i];
    const float a = v[i], b = v[i + HEAD_DIM / 2];
    v[i] = a * c - b * s;
    v[i + HEAD_DIM / 2] = b * c + a * s;
}

// The key/value cache of every prompt in one array per layer: prompt p's position j is slot
// start[p] + j. Writes each row's k and v into its slot. Grid: rows, block: KV_DIM.
__global__ void store_kv_kernel(const float *qkv, const int *slot, float *kc, float *vc)
{
    const int r = blockIdx.x, c = threadIdx.x;
    kc[size_t(slot[r]) * KV_DIM + c] = qkv[size_t(r) * QKV_DIM + Q_DIM + c];
    vc[size_t(slot[r]) * KV_DIM + c] = qkv[size_t(r) * QKV_DIM + Q_DIM + KV_DIM + c];
}

// Prefill attention, tiled as FlashAttention-2 (Dao 2023) does it and as the FP32 CUDA-core
// version flash-attention-minimal (tspeterkim) lays it out: a block keeps a tile of queries in
// shared memory, streams the keys and values through shared memory 32 positions at a time, and
// keeps a running maximum, softmax mass and output per query (online softmax), so each key and
// value is read once per query tile instead of once per query. Nsight Systems showed the
// one-thread-per-(row, head) kernel at 3.23 s of case 03's 7.16 s prefill, on GPU 0 alone.
//
// A block takes FA_ROWS consecutive rows and one key/value head, so its 64 query vectors (16 rows
// x the 4 query heads that read that key/value head) share every K/V tile. Warp w owns query
// vectors 8w .. 8w+7 (two rows): for the scores lane j takes key j of the tile, for the output it
// takes dimensions 4j .. 4j+3. Rows of a block may belong to different prompts; each key is masked
// to its row's own cache, start[r] .. slot[r], and the block walks the union of those ranges.
constexpr int FA_ROWS = 16, FA_Q = FA_ROWS * (N_HEAD / N_KV), FA_KEYS = 32, FA_THREADS = 256;
constexpr int FA_KPAD = HEAD_DIM + 4;   // float4 reads of 8 lanes x key rows fall in distinct banks
constexpr size_t FA_SMEM = sizeof(float) * (size_t(FA_Q) * HEAD_DIM + FA_KEYS * FA_KPAD + FA_KEYS * HEAD_DIM + FA_Q * FA_KEYS);
static_assert(FA_Q == 64 && FA_THREADS / 32 * 8 == FA_Q && HEAD_DIM == 4 * 32 && FA_KEYS == 32, "warp/lane mapping");

__global__ void __launch_bounds__(FA_THREADS) flash_prefill_kernel(const float *qkv, const int *start, const int *slot,
                                                                   const float *kc, const float *vc, float *att, int rows)
{
    extern __shared__ __align__(16) float fa_smem[];
    float *Qs = fa_smem;                          // [FA_Q][HEAD_DIM]
    float *Ks = Qs + FA_Q * HEAD_DIM;             // [FA_KEYS][FA_KPAD]
    float *Vs = Ks + FA_KEYS * FA_KPAD;           // [FA_KEYS][HEAD_DIM]
    float *Ps = Vs + FA_KEYS * HEAD_DIM;          // [FA_Q][FA_KEYS]
    __shared__ int lo[FA_ROWS], hi[FA_ROWS];
    const int r0 = blockIdx.x * FA_ROWS, g = blockIdx.y, t = threadIdx.x, lane = t % 32, warp = t / 32;
    if (t < FA_ROWS) {
        lo[t] = r0 + t < rows ? start[r0 + t] : INT_MAX;   // a row past the batch matches no key
        hi[t] = r0 + t < rows ? slot[r0 + t] : -1;
    }
    for (int i = t; i < FA_Q * HEAD_DIM / 4; i += FA_THREADS) {   // query vector qv = (row qv/4, head 4g + qv%4)
        const int qv = i / (HEAD_DIM / 4), d = i % (HEAD_DIM / 4) * 4, row = r0 + qv / 4;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row < rows) v = *reinterpret_cast<const float4 *>(qkv + size_t(row) * QKV_DIM + (g * 4 + qv % 4) * HEAD_DIM + d);
        *reinterpret_cast<float4 *>(Qs + qv * HEAD_DIM + d) = v;
    }
    __syncthreads();
    int kmin = INT_MAX, kmax = -1;
    for (int i = 0; i < FA_ROWS; i++) { kmin = min(kmin, lo[i]); kmax = max(kmax, hi[i]); }

    float m[8], l[8], o[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        m[i] = -INFINITY; l[i] = 0.f;
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }
    for (int k0 = kmin; k0 <= kmax; k0 += FA_KEYS) {
        __syncthreads();   // every warp is done with the previous tile
        for (int i = t; i < FA_KEYS * HEAD_DIM / 4; i += FA_THREADS) {
            const int j = i / (HEAD_DIM / 4), d = i % (HEAD_DIM / 4) * 4, s = k0 + j;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (s <= kmax) {
                kv = *reinterpret_cast<const float4 *>(kc + size_t(s) * KV_DIM + g * HEAD_DIM + d);
                vv = *reinterpret_cast<const float4 *>(vc + size_t(s) * KV_DIM + g * HEAD_DIM + d);
            }
            *reinterpret_cast<float4 *>(Ks + j * FA_KPAD + d) = kv;
            *reinterpret_cast<float4 *>(Vs + j * HEAD_DIM + d) = vv;
        }
        __syncthreads();
        float s[8] = {};
        for (int d = 0; d < HEAD_DIM; d += 4) {
            const float4 kf = *reinterpret_cast<const float4 *>(Ks + lane * FA_KPAD + d);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const float4 qf = *reinterpret_cast<const float4 *>(Qs + (warp * 8 + i) * HEAD_DIM + d);
                s[i] = fmaf(qf.x, kf.x, s[i]); s[i] = fmaf(qf.y, kf.y, s[i]);
                s[i] = fmaf(qf.z, kf.z, s[i]); s[i] = fmaf(qf.w, kf.w, s[i]);
            }
        }
        const int key = k0 + lane;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int row = (warp * 8 + i) / 4;
            const float sc = key >= lo[row] && key <= hi[row] ? s[i] * ATTN_SCALE : -INFINITY;
            float mx = sc;
            for (int d = 16; d; d /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
            const float mnew = fmaxf(m[i], mx);
            float p = 0.f, alpha = 1.f;
            if (mnew != -INFINITY) {   // some key of this row seen so far
                p = expf(sc - mnew);
                alpha = expf(m[i] - mnew);
            }
            float ps = p;
            for (int d = 16; d; d /= 2) ps += __shfl_xor_sync(0xffffffffu, ps, d);
            l[i] = l[i] * alpha + ps;
            m[i] = mnew;
            Ps[(warp * 8 + i) * FA_KEYS + lane] = p;
            o[i][0] *= alpha; o[i][1] *= alpha; o[i][2] *= alpha; o[i][3] *= alpha;
        }
        __syncwarp();
        for (int j = 0; j < FA_KEYS; j++) {
            const float4 vf = *reinterpret_cast<const float4 *>(Vs + j * HEAD_DIM + lane * 4);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const float p = Ps[(warp * 8 + i) * FA_KEYS + j];
                o[i][0] = fmaf(p, vf.x, o[i][0]); o[i][1] = fmaf(p, vf.y, o[i][1]);
                o[i][2] = fmaf(p, vf.z, o[i][2]); o[i][3] = fmaf(p, vf.w, o[i][3]);
            }
        }
        __syncwarp();   // Ps is rewritten by this warp for the next tile
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int qv = warp * 8 + i, row = r0 + qv / 4;
        if (row >= rows) continue;
        const float inv = 1.f / l[i];
        *reinterpret_cast<float4 *>(att + size_t(row) * Q_DIM + (g * 4 + qv % 4) * HEAD_DIM + lane * 4) =
            make_float4(o[i][0] * inv, o[i][1] * inv, o[i][2] * inv, o[i][3] * inv);
    }
}

// Shared prompt prefixes: pair i copies the key and value of cache slot src[i] to slot dst[i] in
// every layer. Grid: (pairs, N_LAYER), block: KV_DIM.
__global__ void copy_kv_kernel(const int *src, const int *dst, float *kc, float *vc, size_t slots)
{
    const size_t layer = size_t(blockIdx.y) * slots * KV_DIM, c = threadIdx.x;
    const size_t from = layer + size_t(src[blockIdx.x]) * KV_DIM + c, to = layer + size_t(dst[blockIdx.x]) * KV_DIM + c;
    kc[to] = kc[from];
    vc[to] = vc[from];
}

// One thread per (row, query head): softmax(q . k_j * ATTN_SCALE) over the slots start .. slot
// of the row's prompt, and the weighted sum of their v.
__global__ void attention_kernel(const float *qkv, const int *start, const int *slot, const float *kc, const float *vc, float *att, int rows)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= rows * N_HEAD) return;
    const int r = id / N_HEAD, hd = id % N_HEAD, kv = hd / (N_HEAD / N_KV);
    const float *q = qkv + size_t(r) * QKV_DIM + hd * HEAD_DIM;
    auto score = [&](int j) {
        const float *k = kc + size_t(j) * KV_DIM + kv * HEAD_DIM;
        float s = 0.f;
        for (int i = 0; i < HEAD_DIM; i++) s += q[i] * k[i];
        return s * ATTN_SCALE;
    };
    float m = -INFINITY;
    for (int j = start[r]; j <= slot[r]; j++) m = fmaxf(m, score(j));
    float out[HEAD_DIM] = {}, z = 0.f;
    for (int j = start[r]; j <= slot[r]; j++) {
        const float p = expf(score(j) - m);
        const float *v = vc + size_t(j) * KV_DIM + kv * HEAD_DIM;
        z += p;
        for (int i = 0; i < HEAD_DIM; i++) out[i] += p * v[i];
    }
    for (int i = 0; i < HEAD_DIM; i++) att[size_t(r) * Q_DIM + hd * HEAD_DIM + i] = out[i] / z;
}

// Flash-Decoding sequence split: one block per (query head, 32 cached positions).
// Each partial stores the unnormalized value vector, maximum score and softmax mass.
constexpr int KV_TILE = 32;
constexpr int PART_WIDTH = HEAD_DIM + 2;
constexpr int MAX_KV_TILES = (N_CTX + KV_TILE - 1) / KV_TILE;
static_assert(MAX_KV_TILES <= HEAD_DIM, "merge loads one tile per thread");
constexpr int MAX_SPLIT_ROWS = 64;
static_assert(MAX_SPLIT_ROWS * N_HEAD * MAX_KV_TILES * PART_WIDTH <= CHUNK * 2 * I,
              "batched partials must fit expert scratch");

__global__ void decode_partial_kernel(const float *qkv, const float *kc, const float *vc,
                                      const int *starts, const int *pos, int single_start,
                                      int single_length, float *partials)
{
    __shared__ float scores[KV_TILE];
    const int row = blockIdx.z, head = blockIdx.y, tile = blockIdx.x;
    const int length = gridDim.z == 1 ? single_length : pos[row] + 1;
    if (tile * KV_TILE >= length) return;  // this whole block is outside the row's cache
    const int first = gridDim.z == 1 ? single_start : starts[row];
    const int t = threadIdx.x, lane = t % 32;
    const int offset = tile * KV_TILE, count = min(KV_TILE, length - offset);
    const size_t base = size_t(first + offset) * KV_DIM + (head / (N_HEAD / N_KV)) * HEAD_DIM;
    const float *q = qkv + size_t(row) * QKV_DIM + head * HEAD_DIM;
    for (int j = t / 32; j < count; j += HEAD_DIM / 32) {
        const float *k = kc + base + j * KV_DIM;
        float dot = 0.f;
        for (int i = lane; i < HEAD_DIM; i += 32) dot = fmaf(q[i], k[i], dot);
        for (int d = 16; d; d /= 2) dot += __shfl_down_sync(0xffffffffu, dot, d);
        if (lane == 0) scores[j] = dot * ATTN_SCALE;
    }
    __syncthreads();
    float *part = partials + ((row * N_HEAD + head) * gridDim.x + tile) * PART_WIDTH;
    if (t < 32) {
        const float score = t < count ? scores[t] : -INFINITY;
        float maximum = score;
        for (int d = 16; d; d /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d));
        float weight = expf(score - maximum);
        scores[t] = weight;
        for (int d = 16; d; d /= 2) weight += __shfl_down_sync(0xffffffffu, weight, d);
        if (t == 0) {
            part[HEAD_DIM] = maximum;
            part[HEAD_DIM + 1] = weight;
        }
    }
    __syncthreads();
    float value = 0.f;
    for (int j = 0; j < count; j++) value = fmaf(scores[j], vc[base + j * KV_DIM + t], value);
    part[t] = value;
}

__global__ void decode_merge_kernel(const float *partials, int tile_stride, const int *pos, float *att)
{
    __shared__ float scale[MAX_KV_TILES];
    const int row = blockIdx.y, head = blockIdx.x, t = threadIdx.x;
    const int tiles = gridDim.y == 1 ? tile_stride : pos[row] / KV_TILE + 1;
    const int row_head = row * N_HEAD + head;
    const float *parts = partials + row_head * tile_stride * PART_WIDTH;
    if (t < tiles) scale[t] = parts[t * PART_WIDTH + HEAD_DIM];
    __syncthreads();
    float maximum = -INFINITY;
    for (int j = 0; j < tiles; j++) maximum = fmaxf(maximum, scale[j]);
    __syncthreads();
    if (t < tiles) scale[t] = expf(scale[t] - maximum);
    __syncthreads();
    float value = 0.f, mass = 0.f;
    for (int j = 0; j < tiles; j++) {
        const float *part = parts + j * PART_WIDTH;
        value = fmaf(scale[j], part[t], value);
        mass = fmaf(scale[j], part[HEAD_DIM + 1], mass);
    }
    att[row_head * HEAD_DIM + t] = value / mass;
}

// One block per token, one warp per expert. Preserve lowest-index ties.
__global__ void router_kernel(const float *h, const __nv_bfloat16 *router_w, int *sel, float *gate)
{
    __shared__ float lg[N_EXPERT];
    const int r = blockIdx.x, expert = threadIdx.x / 32, lane = threadIdx.x % 32;
    float sum = 0.f;
    for (int i = lane; i < D; i += 32)
        sum = fmaf(h[size_t(r) * D + i], __bfloat162float(router_w[size_t(expert) * D + i]), sum);
    for (int step = 16; step; step /= 2) sum += __shfl_down_sync(0xffffffffu, sum, step);
    if (lane == 0) lg[expert] = sum;
    __syncthreads();
    if (threadIdx.x != 0) return;
    int e1 = 0;
    for (int e = 1; e < N_EXPERT; e++)
        if (lg[e] > lg[e1]) e1 = e;
    int e2 = e1 == 0 ? 1 : 0;
    for (int e = 0; e < N_EXPERT; e++)
        if (e != e1 && lg[e] > lg[e2]) e2 = e;
    const float w2 = expf(lg[e2] - lg[e1]);
    sel[r * TOP_K] = e1;
    sel[r * TOP_K + 1] = e2;
    gate[r * TOP_K] = 1.f / (1.f + w2);
    gate[r * TOP_K + 1] = w2 / (1.f + w2);
}

// Indices address either rows directly or the flattened [row][top-k] routing table.
template <int INDEX_STRIDE = 1>
__global__ void gather_kernel(const float *src, const int *idx, float *dst)
{
    const int i = blockIdx.x;
    const int row = idx[i] / INDEX_STRIDE;
    for (int c = threadIdx.x; c < D; c += blockDim.x) dst[size_t(i) * D + c] = src[size_t(row) * D + c];
}

__global__ void scatter_add_kernel(float *x, const int *routes, const float *gate, const float *y)
{
    const int i = blockIdx.x;
    const int route = routes[i], row = route / TOP_K;
    const float weight = gate[route];
    for (int c = threadIdx.x; c < D; c += blockDim.x) x[size_t(row) * D + c] += weight * y[size_t(i) * D + c];
}

// u[r][j] = silu(t[r][j]) * t[r][I + j], over n = rows * I values.
__global__ void silu_kernel(const float *t, float *u, size_t n)
{
    const size_t k = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (k >= n) return;
    const size_t r = k / I, j = k % I;
    const float gate = t[r * 2 * I + j];
    u[k] = gate / (1.f + expf(-gate)) * t[r * 2 * I + I + j];
}

// One block of 256 per row: the index of the largest logit, the smallest on a tie. Each thread
// scans a strided slice in increasing v, keeping the first maximum; the block then keeps, of two
// candidates, the larger value or on equal values the smaller index, which is the same answer.
constexpr int ARGMAX_THREADS = 256;
__global__ void argmax_kernel(const float *logits, int *next)
{
    __shared__ float best_v[ARGMAX_THREADS];
    __shared__ int best_i[ARGMAX_THREADS];
    const int r = blockIdx.x, t = threadIdx.x;
    const float *row = logits + size_t(r) * VOCAB;
    float bv = row[t];
    int bi = t;
    for (int v = t + ARGMAX_THREADS; v < VOCAB; v += ARGMAX_THREADS)
        if (row[v] > bv) { bv = row[v]; bi = v; }
    best_v[t] = bv;
    best_i[t] = bi;
    __syncthreads();
    for (int half = ARGMAX_THREADS / 2; half > 0; half /= 2) {
        if (t < half) {
            const float ov = best_v[t + half];
            const int oi = best_i[t + half];
            if (ov > best_v[t] || (ov == best_v[t] && oi < best_i[t])) { best_v[t] = ov; best_i[t] = oi; }
        }
        __syncthreads();
    }
    if (t == 0) next[r] = best_i[0];
}

struct Row {
    int prompt, pos, token;
    bool output;   // the prompt's last prompt row, or a decode row: its next token is wanted
};

// ---- per-section timing, only with -DHW5_PROF (the judge's build has none of it) ----------------
// mark(s) waits for both GPUs and charges the time since the previous mark to section s of the
// current phase. Every mark synchronizes, so the sections add up to more than the judge's time.
#ifdef HW5_PROF
enum Seg { S_SETUP, S_EMBED, S_QKV, S_ROPE_KV, S_ATTN, S_OPROJ, S_ROUTER, S_MOE_HOST, S_MOE_XFER,
           S_W1, S_SILU, S_W2, S_SCATTER, S_PARALLEL_MOE, S_LMHEAD, S_OUT_D2H, S_FREE, N_SEG };
const char *const SEG_NAME[N_SEG] = {"setup (alloc, rope)", "embed", "attn_norm + qkv", "rope + store_kv",
                                     "attention", "o_proj", "mlp_norm + router + ids D2H", "moe group + routes H2D",
                                     "moe gather + peer copies", "expert w1", "expert silu", "expert w2",
                                     "expert scatter_add", "two-GPU MoE wall time", "final norm + lm_head", "argmax + D2H", "free"};
struct Prof {
    double t[2][N_SEG] = {};             // [0] prefill, [1] decode
    double flop[2][N_SEG] = {};          // the GEMM sections' arithmetic, for GFLOP/s
    long long expert_rows[2][N_EXPERT] = {};
    long long attn_pairs[2] = {};        // sum over rows of the positions attended to
    long long rows[2] = {};
    int passes[2] = {};
    int phase = 0;
    std::chrono::steady_clock::time_point last;
    std::vector<int> devs;

    void sync()
    {
        int cur;
        cudaGetDevice(&cur);
        for (int d : devs) {
            cudaSetDevice(d);
            check(cudaDeviceSynchronize(), "prof sync");
        }
        cudaSetDevice(cur);
    }
    void start() { sync(); last = std::chrono::steady_clock::now(); }
    void mark(Seg s, double fl = 0)
    {
        sync();
        const auto now = std::chrono::steady_clock::now();
        t[phase][s] += std::chrono::duration<double>(now - last).count();
        flop[phase][s] += fl;
        last = now;
    }
    void report() const
    {
        const char *ph[2] = {"prefill", "decode"};
        double all = 0;
        for (int p = 0; p < 2; p++)
            for (int s = 0; s < N_SEG; s++) all += t[p][s];
        std::fprintf(stderr, "[prof] total %.3f s (synchronized after every section)\n", all);
        for (int p = 0; p < 2; p++) {
            double sum = 0;
            for (int s = 0; s < N_SEG; s++) sum += t[p][s];
            std::fprintf(stderr, "[prof] %s: %.3f s, %d passes, %lld rows, attention pairs %lld\n", ph[p], sum, passes[p],
                         rows[p], attn_pairs[p]);
            for (int s = 0; s < N_SEG; s++) {
                if (t[p][s] == 0) continue;
                std::fprintf(stderr, "[prof]   %-26s %9.3f s %5.1f%%", SEG_NAME[s], t[p][s], 100 * t[p][s] / all);
                if (flop[p][s] > 0) std::fprintf(stderr, "  %8.1f GFLOP/s", flop[p][s] / t[p][s] / 1e9);
                std::fprintf(stderr, "\n");
            }
            std::fprintf(stderr, "[prof]   expert rows (all layers):");
            for (int e = 0; e < N_EXPERT; e++) std::fprintf(stderr, " %lld", expert_rows[p][e]);
            std::fprintf(stderr, "\n");
        }
    }
};
#define PROF(...) prof.__VA_ARGS__
#else
#define PROF(...) ((void)0)
#endif

}  // namespace

void run(const std::vector<Device> &devices, const std::vector<std::vector<int>> &prompts,
         const std::vector<int> &n_gen, std::vector<std::vector<int>> &tokens_out, std::vector<float> &hidden_out)
{
    const int N = int(prompts.size());
    const Device &d0 = devices[0];
#ifdef HW5_NSIGHT
    check(cudaProfilerStart(), "start Nsight capture");
#endif
#ifdef HW5_PROF
    Prof prof;
    for (const Device &dv : devices) prof.devs.push_back(dv.device);
    prof.start();
#endif
    std::vector<int> expert_device(N_EXPERT);   // which Device holds expert e
    for (size_t i = 0; i < devices.size(); i++)
        for (int e = devices[i].first_expert; e < devices[i].first_expert + devices[i].n_expert; e++) expert_device[e] = int(i);

    // On GPU 0: one batch's activations, and every prompt's key/value cache.
    check(cudaSetDevice(d0.device), "cudaSetDevice");
    check(cudaFuncSetAttribute(flash_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(FA_SMEM)),
          "prefill attention shared memory");
    float *x = dalloc<float>(size_t(CHUNK) * D), *h = dalloc<float>(size_t(CHUNK) * D);
    float *qkv = dalloc<float>(size_t(CHUNK) * QKV_DIM), *att = dalloc<float>(size_t(CHUNK) * Q_DIM);
    float *logits = dalloc<float>(size_t(N) * VOCAB);
    int *d_token = dalloc<int>(CHUNK), *d_pos = dalloc<int>(CHUNK), *d_start = dalloc<int>(CHUNK), *d_slot = dalloc<int>(CHUNK);
    int *d_sel = dalloc<int>(size_t(CHUNK) * TOP_K), *d_idx = dalloc<int>(size_t(CHUNK) * TOP_K), *d_next = dalloc<int>(N);
    float *d_gate = dalloc<float>(size_t(CHUNK) * TOP_K);

    std::vector<int> start(N);   // prompt i's position 0 is cache slot start[i]
    size_t slots = 0;
    for (int i = 0; i < N; i++) {
        start[i] = int(slots);
        slots += prompts[i].size() + n_gen[i] - 1;
    }
    float *kc = dalloc<float>(N_LAYER * slots * KV_DIM), *vc = dalloc<float>(N_LAYER * slots * KV_DIM);

    // rope: cos and sin of position * inv_freq[i], [N_CTX][64].
    const std::vector<double> inv = rope_inv_freq();
    std::vector<float> rcos(size_t(N_CTX) * (HEAD_DIM / 2)), rsin(rcos.size());
    for (int p = 0; p < N_CTX; p++)
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            rcos[size_t(p) * (HEAD_DIM / 2) + i] = float(std::cos(p * inv[i]));
            rsin[size_t(p) * (HEAD_DIM / 2) + i] = float(std::sin(p * inv[i]));
        }
    float *d_cos = dalloc<float>(rcos.size()), *d_sin = dalloc<float>(rsin.size());
    check(cudaMemcpy(d_cos, rcos.data(), rcos.size() * 4, cudaMemcpyHostToDevice), "rope");
    check(cudaMemcpy(d_sin, rsin.data(), rsin.size() * 4, cudaMemcpyHostToDevice), "rope");

    // Only undo access enabled by this call. A staged link can still overlap expert work.
    bool peer_owned[2] = {};
    bool parallel = devices.size() == 2;
#ifdef HW5_SERIAL_EXPERTS
    parallel = false;  // diagnostic control: new compute kernels, original serial transport
#endif
    bool direct = parallel;
    for (int i = 0; direct && i < 2; ++i) {
        int can_access = 0;
        const int peer = devices[1 - i].device;
        check(cudaDeviceCanAccessPeer(&can_access, devices[i].device, peer), "query peer access");
        if (!can_access) { direct = false; break; }
        check(cudaSetDevice(devices[i].device), "select peer device");
        cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
        if (status == cudaSuccess) peer_owned[i] = true;
        else if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else if (status == cudaErrorTooManyPeers) { cudaGetLastError(); direct = false; }
        else check(status, "enable peer access");
    }
    cudaEvent_t packed_ready = nullptr, results_ready = nullptr;
    cudaStream_t remote_stream = nullptr;
    float *stage = nullptr, *remote_rows = nullptr, *host_stage = nullptr;
#if defined(HW5_PROF) || defined(HW5_NSIGHT)
    std::fprintf(stderr, "[hw5] expert transport: %s\n",
                 !parallel ? "serial peer copies" : direct ? "packed direct P2P" : "packed host staging");
#endif
    if (parallel) {
        const size_t capacity = std::min(size_t(CHUNK), slots) * TOP_K * D;
        check(cudaSetDevice(d0.device), "select staging device");
        stage = dalloc<float>(capacity);
        if (!direct)
            check(cudaMallocHost(&host_stage, capacity * sizeof(float)), "allocate pinned staging");
        check(cudaEventCreateWithFlags(&packed_ready, cudaEventDisableTiming), "packed event");
        check(cudaSetDevice(devices[1].device), "select remote device");
        remote_rows = dalloc<float>(capacity);
        check(cudaStreamCreateWithFlags(&remote_stream, cudaStreamNonBlocking), "remote stream");
        check(cudaEventCreateWithFlags(&results_ready, cudaEventDisableTiming), "results event");
    }

    // On each GPU: an expert's rows in (he), its hidden layer (t, u) and its output (y).
    struct Scratch { float *he, *t, *u, *y; };
    std::vector<Scratch> scratch(devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
        check(cudaSetDevice(devices[i].device), "cudaSetDevice");
        scratch[i] = {dalloc<float>(size_t(CHUNK) * D), dalloc<float>(size_t(CHUNK) * 2 * I),
                      dalloc<float>(size_t(CHUNK) * I), dalloc<float>(size_t(CHUNK) * D)};
    }
    check(cudaSetDevice(d0.device), "cudaSetDevice");
    PROF(mark(S_SETUP));

    auto upload =[](int *d, const std::vector<int> &v) {
        check(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "upload");
    };

    // One forward pass over up to CHUNK rows. Returns the next token of every output row (-1 for
    // the others) and, in the prefill, writes their hidden state into hidden_out.
    auto forward = [&](const std::vector<Row> &rows, bool prefill) {
#ifdef HW5_NSIGHT
        nvtxRangePushA(prefill ? "prefill" : "decode");
#endif
        const int R = int(rows.size());
        std::vector<int> token(R), pos(R), rstart(R), slot(R);
        std::vector<int> sel(size_t(R) * TOP_K), routes(sel.size());
        for (int r = 0; r < R; r++) {
            token[r] = rows[r].token;
            pos[r] = rows[r].pos;
            rstart[r] = start[rows[r].prompt];
            slot[r] = start[rows[r].prompt] + rows[r].pos;
        }
        upload(d_token, token);
        upload(d_pos, pos);
        upload(d_start, rstart);
        upload(d_slot, slot);
        const int tile_stride = !prefill && R <= MAX_SPLIT_ROWS
            ? (*std::max_element(pos.begin(), pos.end()) / KV_TILE + 1) : 0;
#ifdef HW5_PROF
        prof.phase = prefill ? 0 : 1;
        prof.passes[prof.phase]++;
        prof.rows[prof.phase] += R;
        for (int r = 0; r < R; r++) prof.attn_pairs[prof.phase] += (long long)(rows[r].pos + 1) * N_LAYER;
#endif

        embed_kernel<<<R, 256>>>(d_token, d0.embed, x);
        PROF(mark(S_EMBED));
        for (int l = 0; l < N_LAYER; l++) {
            const LayerWeights &L = d0.layers[l];
            float *k_l = kc + size_t(l) * slots * KV_DIM, *v_l = vc + size_t(l) * slots * KV_DIM;

            // Attention.
            rmsnorm_kernel<<<R, 256>>>(x, L.attn_norm, h);
            linear(h, L.qkv_w, qkv, R, QKV_DIM, D);
            PROF(mark(S_QKV, 2.0 * R * QKV_DIM * D));
            rope_kernel<<<dim3(R, N_HEAD + N_KV), HEAD_DIM / 2>>>(qkv, d_pos, d_cos, d_sin);
            store_kv_kernel<<<R, KV_DIM>>>(qkv, d_slot, k_l, v_l);
            PROF(mark(S_ROPE_KV));
            if (tile_stride) {
                // The expert intermediate is idle until the MoE stage on this same stream.
                float *parts = scratch[0].t;
                decode_partial_kernel<<<dim3(tile_stride, N_HEAD, R), HEAD_DIM>>>(
                    qkv, k_l, v_l, d_start, d_pos, rstart[0], pos[0] + 1, parts);
                decode_merge_kernel<<<dim3(N_HEAD, R), HEAD_DIM>>>(parts, tile_stride, d_pos, att);
            } else if (prefill) {
                flash_prefill_kernel<<<dim3(blocks(R, FA_ROWS), N_KV), FA_THREADS, FA_SMEM>>>(qkv, d_start, d_slot, k_l, v_l, att, R);
            } else {   // a decode batch of more than MAX_SPLIT_ROWS rows, one per prompt
                attention_kernel<<<blocks(size_t(R) * N_HEAD, 128), 128>>>(qkv, d_start, d_slot, k_l, v_l, att, R);
            }
            PROF(mark(S_ATTN));
            linear(att, L.o_w, x, R, D, Q_DIM, true);
            PROF(mark(S_OPROJ, 2.0 * R * D * Q_DIM));

            // Group flat routing-table indices by expert. Gates stay on GPU 0 for scatter.
            rmsnorm_kernel<<<R, 256>>>(x, L.mlp_norm, h);
            router_kernel<<<R, N_EXPERT * 32>>>(h, L.router_w, d_sel, d_gate);
            check(cudaMemcpy(sel.data(), d_sel, sel.size() * 4, cudaMemcpyDeviceToHost), "router");
            PROF(mark(S_ROUTER));
            int offsets[N_EXPERT + 1] = {};
            for (int e : sel) ++offsets[e + 1];
            for (int e = 0; e < N_EXPERT; ++e) offsets[e + 1] += offsets[e];
            int cursor[N_EXPERT];
            std::copy_n(offsets, N_EXPERT, cursor);
            for (int j = 0; j < R * TOP_K; ++j) routes[cursor[sel[j]]++] = j;
            upload(d_idx, routes);
            PROF(mark(S_MOE_HOST));
            // A contiguous expert range on each device makes each remote batch one transfer.
            const int split = parallel ? devices[1].first_expert : N_EXPERT;
            const int first_remote = offsets[split];
            const int n_remote = offsets[N_EXPERT] - first_remote;
            auto project_expert = [&](int e, int di, int n, const float *input, float *output, cudaStream_t stream) {
                const Device &device = devices[di];
                const Scratch &work = scratch[di];
                const size_t j = size_t(e - device.first_expert);
                linear(input, device.layers[l].w1 + j * 2 * I * D, work.t, n, 2 * I, D, false, stream);
                if (!n_remote) { PROF(mark(S_W1, 2.0 * n * 2 * I * D)); }
                silu_kernel<<<blocks(size_t(n) * I, 256), 256, 0, stream>>>(work.t, work.u, size_t(n) * I);
                if (!n_remote) { PROF(mark(S_SILU)); }
                linear(work.u, device.layers[l].w2 + j * D * I, output, n, D, I, false, stream);
                if (!n_remote) { PROF(mark(S_W2, 2.0 * n * D * I)); }
#ifdef HW5_PROF
                prof.expert_rows[prof.phase][e] += n;
#endif
            };
            if (n_remote) {
                const Device &remote = devices[1];
                const size_t bytes = size_t(n_remote) * D * sizeof(float);
                gather_kernel<TOP_K><<<n_remote, 256>>>(h, d_idx + first_remote, stage);
                if (direct)
                    check(cudaMemcpyPeerAsync(remote_rows, remote.device, stage, d0.device, bytes), "send packed rows");
                else
                    check(cudaMemcpyAsync(host_stage, stage, bytes, cudaMemcpyDeviceToHost), "stage remote inputs");
                check(cudaEventRecord(packed_ready), "publish remote inputs");
                check(cudaSetDevice(remote.device), "select remote experts");
                check(cudaStreamWaitEvent(remote_stream, packed_ready, 0), "wait for gathered rows");
                if (!direct)
                    check(cudaMemcpyAsync(remote_rows, host_stage, bytes, cudaMemcpyHostToDevice, remote_stream), "load remote inputs");
                for (int e = split; e < N_EXPERT; ++e) {
                    const int n = offsets[e + 1] - offsets[e];
                    if (!n) continue;
                    float *rows_e = remote_rows + size_t(offsets[e] - first_remote) * D;
                    project_expert(e, 1, n, rows_e, rows_e, remote_stream);
                }
                if (direct)
                    check(cudaMemcpyPeerAsync(stage, d0.device, remote_rows, remote.device, bytes, remote_stream), "return packed rows");
                else
                    check(cudaMemcpyAsync(host_stage, remote_rows, bytes, cudaMemcpyDeviceToHost, remote_stream), "stage remote results");
                check(cudaEventRecord(results_ready, remote_stream), "publish remote results");
                check(cudaSetDevice(d0.device), "select local experts");
            }
            for (int e = 0; e < split; e++) {
                const int n = offsets[e + 1] - offsets[e];
                if (n == 0) continue;
                const int di = expert_device[e];
                const Device &dv = devices[di];
                const Scratch &S0 = scratch[0], &S = scratch[di];
                const int *expert_routes = d_idx + offsets[e];
                gather_kernel<TOP_K><<<n, 256>>>(h, expert_routes, S0.he);
                if (di != 0) {
                    check(cudaMemcpyPeer(S.he, dv.device, S0.he, d0.device, size_t(n) * D * 4), "rows to the expert's GPU");
                    check(cudaSetDevice(dv.device), "cudaSetDevice");
                }
                if (!n_remote) { PROF(mark(S_MOE_XFER)); }
                project_expert(e, di, n, S.he, S.y, nullptr);
                if (di != 0) {
                    check(cudaMemcpyPeer(S0.y, d0.device, S.y, dv.device, size_t(n) * D * 4), "rows back");
                    check(cudaSetDevice(d0.device), "cudaSetDevice");
                    if (!n_remote) { PROF(mark(S_MOE_XFER)); }
                }
                scatter_add_kernel<<<n, 256>>>(x, expert_routes, d_gate, S0.y);
                if (!n_remote) { PROF(mark(S_SCATTER)); }
            }
            if (n_remote) {
                check(cudaStreamWaitEvent(nullptr, results_ready, 0), "join remote experts");
                if (!direct)
                    check(cudaMemcpyAsync(stage, host_stage, size_t(n_remote) * D * sizeof(float),
                                          cudaMemcpyHostToDevice), "load remote results");
                // Preserve the original expert-ID accumulation order for rows routed twice.
                for (int e = split; e < N_EXPERT; ++e) {
                    const int n = offsets[e + 1] - offsets[e];
                    if (n) scatter_add_kernel<<<n, 256>>>(x, d_idx + offsets[e], d_gate,
                        stage + size_t(offsets[e] - first_remote) * D);
                }
                PROF(mark(S_PARALLEL_MOE));
            }
        }

        // The output rows: their hidden state and next token.
        std::vector<int> out;
        for (int r = 0; r < R; r++)
            if (rows[r].output) out.push_back(r);
        std::vector<int> next(R, -1);
        const int n = int(out.size());
        if (n == 0) {
#ifdef HW5_NSIGHT
            nvtxRangePop();
#endif
            return next;
        }
        upload(d_idx, out);
        gather_kernel<1><<<n, 256>>>(x, d_idx, att);   // att holds the output rows from here
        rmsnorm_kernel<<<n, 256>>>(att, d0.final_norm, h);
        linear(h, d0.lm_head, logits, n, VOCAB, D);
        PROF(mark(S_LMHEAD, 2.0 * n * VOCAB * D));
        argmax_kernel<<<n, ARGMAX_THREADS>>>(logits, d_next);
        std::vector<int> best(n);
        check(cudaMemcpy(best.data(), d_next, n * 4, cudaMemcpyDeviceToHost), "next tokens");
        for (int o = 0; o < n; o++) next[out[o]] = best[o];
        if (prefill)
            for (int o = 0; o < n; o++)
                check(cudaMemcpy(&hidden_out[size_t(rows[out[o]].prompt) * D], h + size_t(o) * D, D * 4, cudaMemcpyDeviceToHost), "hidden");
        PROF(mark(S_OUT_D2H));
#ifdef HW5_NSIGHT
        nvtxRangePop();
#endif
        return next;
    };

    // Shared prefixes, as in vLLM's automatic prefix caching and SGLang's RadixAttention: the key
    // and value of a position depend only on the tokens up to it, so prompts that begin with the
    // same tokens can compute them once. The host only compares token ids, choosing which rows to
    // run (Section 2.5 lets it decide the batches); the GPU computes every value it then copies.
    //
    // Prompts are taken longest first. Each is either a donor, all of whose rows run in the first
    // round, or shares its longest common prefix with one donor: after the first round that prefix's
    // cache slots are copied from the donor in every layer, and only the prompt's later rows run in
    // the second round. A prompt always runs its own last row, which gives its hidden state and
    // first token, so it shares at most L - 1 positions. A prompt shares only a prefix of at least
    // MIN_SHARED_PREFIX tokens (every prompt begins with token 1), and sharing is skipped when it
    // would save fewer than MIN_SHARED_ROWS rows in all, below the cost of the second round's pass.
    constexpr int MIN_SHARED_PREFIX = 8, MIN_SHARED_ROWS = 128;
    std::vector<int> order(N), shared(N, 0), donor(N, -1), donors;
    for (int i = 0; i < N; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return prompts[a].size() > prompts[b].size(); });
    long long saved = 0;
    for (int i : order) {
        const std::vector<int> &p = prompts[i];
        int best = 0, from = -1;
        for (int d : donors) {
            const std::vector<int> &q = prompts[d];
            const size_t n = std::min(p.size(), q.size());
            size_t lcp = 0;
            while (lcp < n && p[lcp] == q[lcp]) lcp++;
            if (int(lcp) > best) { best = int(lcp); from = d; }
        }
        best = std::min(best, int(p.size()) - 1);
        if (best >= MIN_SHARED_PREFIX) { shared[i] = best; donor[i] = from; saved += best; }
        else donors.push_back(i);
    }
    if (saved < MIN_SHARED_ROWS) {
        std::fill(shared.begin(), shared.end(), 0);
        std::fill(donor.begin(), donor.end(), -1);
    }

    // Prefill: each round's rows in prompt order, CHUNK rows at a time. A batch's rows read cache
    // slots that earlier batches, the batch itself, or the copy before the round wrote.
    std::vector<int> last(N), len(N);   // each prompt's last token and its positions so far
    auto prefill_round = [&](bool sharers) {
        std::vector<Row> all;
        for (int i = 0; i < N; i++) {
            if ((donor[i] >= 0) != sharers) continue;
            const int L = int(prompts[i].size());
            for (int p = shared[i]; p < L; p++) all.push_back({i, p, prompts[i][p], p == L - 1});
        }
        for (size_t c = 0; c < all.size(); c += CHUNK) {
            const std::vector<Row> rows(all.begin() + c, all.begin() + std::min(all.size(), c + CHUNK));
            const std::vector<int> next = forward(rows, true);
            for (size_t r = 0; r < rows.size(); r++)
                if (rows[r].output) last[rows[r].prompt] = tokens_out[rows[r].prompt][0] = next[r];
        }
    };
    prefill_round(false);
    if (saved >= MIN_SHARED_ROWS) {
        std::vector<int> src, dst;
        for (int i = 0; i < N; i++)
            for (int p = 0; p < shared[i]; p++) {
                src.push_back(start[donor[i]] + p);
                dst.push_back(start[i] + p);
            }
        int *d_src = dalloc<int>(src.size()), *d_dst = dalloc<int>(dst.size());
        upload(d_src, src);
        upload(d_dst, dst);
        copy_kv_kernel<<<dim3(unsigned(src.size()), N_LAYER), KV_DIM>>>(d_src, d_dst, kc, vc, slots);
        check(cudaGetLastError(), "copy shared prefixes");
        prefill_round(true);
        check(cudaFree(d_src), "cudaFree");
        check(cudaFree(d_dst), "cudaFree");
    }
    for (int i = 0; i < N; i++) len[i] = int(prompts[i].size());

    // Decode: one row per prompt still generating, step by step.
    int max_k = 0;
    for (int k : n_gen) max_k = std::max(max_k, k);
    for (int t = 1; t < max_k; t++) {
        std::vector<Row> rows;
        for (int i = 0; i < N; i++)
            if (t < n_gen[i]) rows.push_back({i, len[i], last[i], true});
        const std::vector<int> next = forward(rows, false);   // N <= 2048 < CHUNK rows
        for (size_t r = 0; r < rows.size(); r++) {
            const int i = rows[r].prompt;
            last[i] = tokens_out[i][t] = next[r];
            len[i]++;
        }
    }

    // Free everything.
    check(cudaSetDevice(d0.device), "cudaSetDevice");
    for (void *p : {(void *)x, (void *)h, (void *)qkv, (void *)att, (void *)logits, (void *)d_token, (void *)d_pos,
                    (void *)d_start, (void *)d_slot, (void *)d_sel, (void *)d_idx, (void *)d_next, (void *)d_gate,
                    (void *)kc, (void *)vc, (void *)d_cos, (void *)d_sin})
        check(cudaFree(p), "cudaFree");
    for (size_t i = 0; i < devices.size(); i++) {
        check(cudaSetDevice(devices[i].device), "cudaSetDevice");
        for (float *p : {scratch[i].he, scratch[i].t, scratch[i].u, scratch[i].y}) check(cudaFree(p), "cudaFree");
    }
    if (parallel) {
        check(cudaSetDevice(devices[1].device), "release remote staging");
        check(cudaFree(remote_rows), "free remote rows");
        check(cudaStreamDestroy(remote_stream), "destroy remote stream");
        check(cudaEventDestroy(results_ready), "destroy results event");
        check(cudaSetDevice(d0.device), "release local staging");
        check(cudaFree(stage), "free staging");
        if (host_stage) check(cudaFreeHost(host_stage), "free pinned staging");
        check(cudaEventDestroy(packed_ready), "destroy packed event");
    }
    for (int i = 0; i < 2; ++i) if (peer_owned[i]) {
        check(cudaSetDevice(devices[i].device), "release peer access");
        check(cudaDeviceDisablePeerAccess(devices[1 - i].device), "disable peer access");
    }
#ifdef HW5_NSIGHT
    check(cudaSetDevice(d0.device), "restore capture device");
    check(cudaProfilerStop(), "stop Nsight capture");
#endif
#ifdef HW5_PROF
    prof.phase = 1;
    prof.mark(S_FREE);
    prof.report();
#endif
}
