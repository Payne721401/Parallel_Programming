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

// Few rows (decode): a projection reads all of W for little arithmetic, so it is bound by how
// fast W streams in, as in TensorRT-LLM's batched GEMV. One warp computes one output column for
// up to R rows: each lane loads 8 bf16 of the column's row of W at once (16 bytes), widens them
// exactly by shifting the bits into an fp32 (bf16 is fp32's top half), and uses each against the
// same 8 values of every row's x (two float4 loads), so W is read once for all R rows. The 8 warps
// of a block share x through L1.
//
// Every product and sum is fp32. Each lane sums its 8-wide chunks in increasing k, and the warp
// adds the 32 lane sums in one fixed tree, the same for every R, so a row's result does not depend
// on how many rows share the launch.
constexpr int SKINNY_WARPS = 8, SKINNY_MAX_ROWS = 64;
static_assert(D % 256 == 0 && Q_DIM % 256 == 0 && I % 256 == 0, "every K is a whole number of 32 lanes x 8");

__device__ __forceinline__ float bf16_lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16_hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }

template <int R>
__global__ void __launch_bounds__(SKINNY_WARPS * 32) skinny_kernel(const float *x, const __nv_bfloat16 *W, float *y,
                                                                   int M, int N, int K, bool add)
{
    const int out = blockIdx.x * SKINNY_WARPS + threadIdx.x / 32;
    if (out >= N) return;   // uniform within each warp
    const int lane = threadIdx.x % 32, first = blockIdx.y * R;
    const uint4 *w = reinterpret_cast<const uint4 *>(W + size_t(out) * K);
    const float4 *xr[R];
    #pragma unroll
    for (int r = 0; r < R; r++) xr[r] = reinterpret_cast<const float4 *>(x + size_t(min(first + r, M - 1)) * K);
    float sum[R];
    #pragma unroll
    for (int r = 0; r < R; r++) sum[r] = 0.f;
    #pragma unroll 2
    for (int c = lane; c < K / 8; c += 32) {
        const uint4 p = __ldg(w + c);
        const float wf[8] = {bf16_lo(p.x), bf16_hi(p.x), bf16_lo(p.y), bf16_hi(p.y),
                             bf16_lo(p.z), bf16_hi(p.z), bf16_lo(p.w), bf16_hi(p.w)};
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const float4 a = __ldg(xr[r] + 2 * c), b = __ldg(xr[r] + 2 * c + 1);
            float s = sum[r];
            s = fmaf(a.x, wf[0], s); s = fmaf(a.y, wf[1], s); s = fmaf(a.z, wf[2], s); s = fmaf(a.w, wf[3], s);
            s = fmaf(b.x, wf[4], s); s = fmaf(b.y, wf[5], s); s = fmaf(b.z, wf[6], s); s = fmaf(b.w, wf[7], s);
            sum[r] = s;
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float s = sum[r];
        for (int d = 16; d; d /= 2) s += __shfl_down_sync(0xffffffffu, s, d);
        if (lane == 0 && first + r < M) {
            float *dst = y + size_t(first + r) * N + out;
            *dst = add ? *dst + s : s;
        }
    }
}

template <int R>
void skinny(const float *x, const __nv_bfloat16 *W, float *y, int M, int N, int K, bool add, cudaStream_t stream)
{
    skinny_kernel<R><<<dim3(blocks(N, SKINNY_WARPS), blocks(M, R)), SKINNY_WARPS * 32, 0, stream>>>(x, W, y, M, N, K, add);
}

// Every K here (D, Q_DIM, I) is a multiple of TK and of 256. x, W and y are the starts of
// allocations or of whole rows in them, so the 16-byte loads of skinny_kernel are aligned.
void linear(const float *x, const __nv_bfloat16 *W, float *y, int M, int N, int K, bool add = false, cudaStream_t stream = nullptr)
{
    if (M <= SKINNY_MAX_ROWS) {   // one launch reads W once per 32 rows at most
        if (M == 1) skinny<1>(x, W, y, M, N, K, add, stream);
        else if (M <= 2) skinny<2>(x, W, y, M, N, K, add, stream);
        else if (M <= 4) skinny<4>(x, W, y, M, N, K, add, stream);
        else if (M <= 8) skinny<8>(x, W, y, M, N, K, add, stream);
        else if (M <= 16) skinny<16>(x, W, y, M, N, K, add, stream);
        else skinny<32>(x, W, y, M, N, K, add, stream);
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
            } else {
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

    // Prefill: every prompt's tokens in order, CHUNK rows at a time. A batch's rows read cache
    // slots that earlier batches or the batch itself wrote.
    std::vector<Row> all;
    for (int i = 0; i < N; i++) {
        const int L = int(prompts[i].size());
        for (int p = 0; p < L; p++) all.push_back({i, p, prompts[i][p], p == L - 1});
    }
    std::vector<int> last(N), len(N);   // each prompt's last token and its positions so far
    for (size_t c = 0; c < all.size(); c += CHUNK) {
        const std::vector<Row> rows(all.begin() + c, all.begin() + std::min(all.size(), c + CHUNK));
        const std::vector<int> next = forward(rows, true);
        for (size_t r = 0; r < rows.size(); r++)
            if (rows[r].output) last[rows[r].prompt] = tokens_out[rows[r].prompt][0] = next[r];
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
