// HW 5: YOUR FILE. Everything here is yours to change, behind the one function main() calls:
// run(). mixtral.hpp says what run() is given and what it must leave behind.
//
// A ROW is one token at one position of one prompt. forward() runs one batch of rows through the
// model: the prefill is every prompt's tokens, CHUNK rows at a time, and each decode step is one
// row per prompt still generating.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
// each thread a 4x4 piece of it, from 64x16 slices of x and W staged in shared memory.
constexpr int TM = 64, TN = 64, TK = 16;
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

// Single-row projection: one warp computes one dot product along a contiguous row of W.
constexpr int GEMV_WARPS = 4;
__global__ void gemv_kernel(const float *x, const __nv_bfloat16 *W, float *y, int N, int K, bool add)
{
    const int out = blockIdx.x * GEMV_WARPS + threadIdx.x / 32;
    if (out >= N) return;  // uniform within each warp
    const int lane = threadIdx.x % 32;
    const __nv_bfloat16 *w = W + size_t(out) * K;
    float sum = 0.f;
    for (int k = lane; k < K; k += 32)
        sum = fmaf(x[k], __bfloat162float(w[k]), sum);
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) y[out] = add ? y[out] + sum : sum;
}

// Share each converted weight across up to four token rows in the same warp.
constexpr int SMALL_ROWS = 4;
__global__ void small_linear_kernel(const float *x, const __nv_bfloat16 *W, float *y,
                                    int M, int N, int K, bool add)
{
    const int out = blockIdx.x * GEMV_WARPS + threadIdx.x / 32;
    if (out >= N) return;
    const int first = blockIdx.y * SMALL_ROWS, lane = threadIdx.x % 32;
    const __nv_bfloat16 *w = W + size_t(out) * K;
    const float *inputs[SMALL_ROWS];
    float sums[SMALL_ROWS] = {};
    #pragma unroll
    for (int r = 0; r < SMALL_ROWS; r++)
        inputs[r] = x + size_t(min(first + r, M - 1)) * K;
    for (int k = lane; k < K; k += 32) {
        const float weight = __bfloat162float(w[k]);
        #pragma unroll
        for (int r = 0; r < SMALL_ROWS; r++)
            sums[r] = fmaf(inputs[r][k], weight, sums[r]);
    }
    #pragma unroll
    for (int r = 0; r < SMALL_ROWS; r++) {
        if (first + r >= M) break;
        float sum = sums[r];
        for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (lane == 0) {
            float *dst = y + size_t(first + r) * N + out;
            *dst = add ? *dst + sum : sum;
        }
    }
}

// Every K here (D, Q_DIM, I) is a multiple of TK.
void linear(const float *x, const __nv_bfloat16 *W, float *y, int M, int N, int K, bool add = false)
{
    if (M == 1) {
        gemv_kernel<<<blocks(N, GEMV_WARPS), GEMV_WARPS * 32>>>(x, W, y, N, K, add);
        return;
    }
    if (M <= 16) {
        dim3 grid(blocks(N, GEMV_WARPS), blocks(M, SMALL_ROWS));
        small_linear_kernel<<<grid, GEMV_WARPS * 32>>>(x, W, y, M, N, K, add);
        return;
    }
    dim3 grid(blocks(N, TN), blocks(M, TM));
    linear_kernel<<<grid, 256>>>(x, W, y, M, N, K, add);
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

// One thread per row: the 8 router logits, the two largest (the lower index on a tie) and their
// softmax weights.
__global__ void router_kernel(const float *h, const __nv_bfloat16 *router_w, int *sel, float *gate, int rows)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float lg[N_EXPERT];
    for (int e = 0; e < N_EXPERT; e++) {
        float s = 0.f;
        for (int i = 0; i < D; i++) s += h[size_t(r) * D + i] * __bfloat162float(router_w[size_t(e) * D + i]);
        lg[e] = s;
    }
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

// One thread per row: the index of the largest logit, the smallest on a tie.
__global__ void argmax_kernel(const float *logits, int *next, int rows)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int best = 0;
    for (int v = 1; v < VOCAB; v++)
        if (logits[size_t(r) * VOCAB + v] > logits[size_t(r) * VOCAB + best]) best = v;
    next[r] = best;
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
           S_W1, S_SILU, S_W2, S_SCATTER, S_LMHEAD, S_OUT_D2H, S_FREE, N_SEG };
const char *const SEG_NAME[N_SEG] = {"setup (alloc, rope)", "embed", "attn_norm + qkv", "rope + store_kv",
                                     "attention", "o_proj", "mlp_norm + router + ids D2H", "moe group + routes H2D",
                                     "moe gather + peer copies", "expert w1", "expert silu", "expert w2",
                                     "expert scatter_add", "final norm + lm_head", "argmax + D2H", "free"};
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
            router_kernel<<<blocks(R, 128), 128>>>(h, L.router_w, d_sel, d_gate, R);
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
            for (int e = 0; e < N_EXPERT; e++) {
                const int n = offsets[e + 1] - offsets[e];
                if (n == 0) continue;
                const int di = expert_device[e];
                const Device &dv = devices[di];
                const Scratch &S0 = scratch[0], &S = scratch[di];
                const size_t j = size_t(e - dv.first_expert);   // its index on its device
                const int *expert_routes = d_idx + offsets[e];
#ifdef HW5_PROF
                prof.expert_rows[prof.phase][e] += n;
#endif
                gather_kernel<TOP_K><<<n, 256>>>(h, expert_routes, S0.he);
                if (di != 0) {
                    check(cudaMemcpyPeer(S.he, dv.device, S0.he, d0.device, size_t(n) * D * 4), "rows to the expert's GPU");
                    check(cudaSetDevice(dv.device), "cudaSetDevice");
                }
                PROF(mark(S_MOE_XFER));
                linear(S.he, dv.layers[l].w1 + j * 2 * I * D, S.t, n, 2 * I, D);
                PROF(mark(S_W1, 2.0 * n * 2 * I * D));
                silu_kernel<<<blocks(size_t(n) * I, 256), 256>>>(S.t, S.u, size_t(n) * I);
                PROF(mark(S_SILU));
                linear(S.u, dv.layers[l].w2 + j * D * I, S.y, n, D, I);
                PROF(mark(S_W2, 2.0 * n * D * I));
                if (di != 0) {
                    check(cudaMemcpyPeer(S0.y, d0.device, S.y, dv.device, size_t(n) * D * 4), "rows back");
                    check(cudaSetDevice(d0.device), "cudaSetDevice");
                    PROF(mark(S_MOE_XFER));
                }
                scatter_add_kernel<<<n, 256>>>(x, expert_routes, d_gate, S0.y);
                PROF(mark(S_SCATTER));
            }
        }

        // The output rows: their hidden state and next token.
        std::vector<int> out;
        for (int r = 0; r < R; r++)
            if (rows[r].output) out.push_back(r);
        std::vector<int> next(R, -1);
        const int n = int(out.size());
        if (n == 0) return next;
        upload(d_idx, out);
        gather_kernel<1><<<n, 256>>>(x, d_idx, att);   // att holds the output rows from here
        rmsnorm_kernel<<<n, 256>>>(att, d0.final_norm, h);
        linear(h, d0.lm_head, logits, n, VOCAB, D);
        PROF(mark(S_LMHEAD, 2.0 * n * VOCAB * D));
        argmax_kernel<<<blocks(n, 128), 128>>>(logits, d_next, n);
        std::vector<int> best(n);
        check(cudaMemcpy(best.data(), d_next, n * 4, cudaMemcpyDeviceToHost), "next tokens");
        for (int o = 0; o < n; o++) next[out[o]] = best[o];
        if (prefill)
            for (int o = 0; o < n; o++)
                check(cudaMemcpy(&hidden_out[size_t(rows[out[o]].prompt) * D], h + size_t(o) * D, D * 4, cudaMemcpyDeviceToHost), "hidden");
        PROF(mark(S_OUT_D2H));
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
#ifdef HW5_PROF
    prof.phase = 1;
    prof.mark(S_FREE);
    prof.report();
#endif
}
