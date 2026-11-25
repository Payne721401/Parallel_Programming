// HW 5: the model, where its weights sit on the two GPUs, and the one function you write.
//
// FIXED FILE. The judge compiles your hw5.cu against its own copies of this file and main.cu,
// so editing yours changes nothing there.
//
// ---- the model: Mixtral-8x7B -----------------------------------------------------------------
//
// A decoder-only transformer with a mixture of experts in every layer. Sizes:
//
//   N_LAYER = 32 layers, D = 4096 the width of the residual stream, VOCAB = 32000 tokens
//   N_HEAD = 32 query heads and N_KV = 8 key/value heads of HEAD_DIM = 128: query head h reads
//     key/value head h / 4
//   N_EXPERT = 8 experts per layer, TOP_K = 2 of them per row, each a two-layer MLP of width
//     I = 14336
//
// A row is one token at one position of one prompt. Running the model over n rows, with x an
// n x D matrix:
//
//   x = embed[token]
//   for each layer l (32 of them):
//       h = RMSNorm(x; attn_norm)
//       q, k, v = h * qkv_w^T                          q: 32 heads of 128, k and v: 8 heads of 128
//       q, k = rope(q, position), rope(k, position)   per head; append k, v to the prompt's cache
//       a = attention of q over the prompt's cache, per head
//       x += a * o_w^T
//       h = RMSNorm(x; mlp_norm)
//       r = h * router_w^T                             8 router logits
//       e_1, e_2 = the 2 experts of largest r          (the lowest index first on a tie)
//       g_1, g_2 = softmax(r[e_1], r[e_2])
//       x += g_1 * expert(e_1, h) + g_2 * expert(e_2, h)
//   hidden = RMSNorm(x; final_norm)                   the output's hidden state
//   next token = argmax over v of  hidden . lm_head[v]   (the smallest v on a tie)
//
// where
//
//   RMSNorm(x; s)[i] = x[i] / sqrt(mean_j(x[j]^2) + RMS_EPS) * s[i]
//
//   rope rotates the pairs (q[i], q[i + 64]), i < 64, of every head by the angle
//   position * inv_freq[i], inv_freq[i] = ROPE_THETA^(-2i/128):
//       (q[i], q[i+64]) <- (q[i] cos t - q[i+64] sin t,  q[i+64] cos t + q[i] sin t)
//   rope_inv_freq() below computes inv_freq; main() and the reference use that function.
//
//   attention, for query head h at position i of a prompt, over the cached positions j <= i of
//   its key/value head h / 4: s_j = q . k_j * ATTN_SCALE, p = softmax over the s_j,
//   a = sum_j p_j v_j.
//
//   expert(e, h): t = h * w1[e]^T, a vector of 2I whose first I entries are the gate and the
//   last I the up projection; u[j] = silu(t[j]) * t[I + j] with silu(z) = z / (1 + e^-z); the
//   expert's output is u * w2[e]^T.
//
// Every matrix W is [out][in], row-major: y[r][o] = sum_i x[r][i] * W[o][i]. No layer has a bias.
//
// A prompt is processed in two phases. In the prefill phase its L tokens are the rows at
// positions 0 .. L-1; the hidden state of the last row is the output `hidden`, and the token
// predicted from it is tokens[0]. In the decode phase tokens[t] is predicted from one row,
// token tokens[t-1] at position L+t-1, attending to every earlier position through the
// prompt's key/value cache; it runs for K-1 steps. Prompts are independent: rows of different
// prompts may share a batch as long as each row attends only to its own prompt's cache, and
// any prompt's rows may go to any expert's GPU.
//
// ---- where the weights are ------------------------------------------------------------------
//
// main() reads the weight file onto the GPUs in the layout below and hands run() one Device
// per GPU, before the clock starts. Every array is bf16 device memory on that device, the
// model's own precision (every bf16 value is an fp32 value: __bfloat162float is exact):
//
//   - the shared weights (embed, every layer's attention, norms and router, final_norm,
//     lm_head), 3 GB, are on EVERY device, a copy each;
//   - the experts are split: device i holds, in every layer, the experts first_expert ..
//     first_expert + n_expert - 1 and no other. With two GPUs that is 0..3 and 4..7, 45 GB
//     each. The model is 93 GB and a GPU has 80, which is why one GPU cannot hold it.
//
// Apart from these arrays the devices' memory is yours: the key/value caches, the batches, any
// other form of the weights and everything else run() needs it allocates itself, and frees
// before it returns. run() may read the weight arrays and copy them anywhere, on the clock,
// and must not free or overwrite them.

#ifndef MIXTRAL_HPP
#define MIXTRAL_HPP

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <vector>

constexpr int N_LAYER = 32;
constexpr int D = 4096;           // the residual stream
constexpr int VOCAB = 32000;
constexpr int N_HEAD = 32;        // query heads
constexpr int N_KV = 8;           // key/value heads; query head h uses key/value head h / (N_HEAD / N_KV)
constexpr int HEAD_DIM = 128;
constexpr int Q_DIM = N_HEAD * HEAD_DIM;      // 4096
constexpr int KV_DIM = N_KV * HEAD_DIM;       // 1024
constexpr int QKV_DIM = Q_DIM + 2 * KV_DIM;   // 6144: qkv_w's rows are all Q heads | all K heads | all V heads
constexpr int N_EXPERT = 8;
constexpr int TOP_K = 2;
constexpr int I = 14336;          // an expert's width: w1 has 2I rows, the gate's I then the up's I
constexpr int N_CTX = 4096;       // L + K - 1 <= N_CTX for every prompt
constexpr float RMS_EPS = 1e-5f;
constexpr float ATTN_SCALE = 0.08838834764831845f;   // 1 / sqrt(HEAD_DIM)
constexpr double ROPE_THETA = 1000000.0;

// inv_freq[i] for i < HEAD_DIM / 2.
inline std::vector<double> rope_inv_freq()
{
    std::vector<double> inv(HEAD_DIM / 2);
    for (int i = 0; i < HEAD_DIM / 2; i++) inv[i] = std::pow(ROPE_THETA, -2.0 * i / HEAD_DIM);
    return inv;
}

// One layer's weights on one device, all bf16.
struct LayerWeights {
    const __nv_bfloat16 *attn_norm;   // [D]
    const __nv_bfloat16 *qkv_w;       // [QKV_DIM][D]
    const __nv_bfloat16 *o_w;         // [D][Q_DIM]
    const __nv_bfloat16 *mlp_norm;    // [D]
    const __nv_bfloat16 *router_w;    // [N_EXPERT][D]
    // The device's experts, n_expert of them: expert first_expert + j is at index j.
    const __nv_bfloat16 *w1;          // [n_expert][2 * I][D]   rows 0 .. I-1: gate, I .. 2I-1: up
    const __nv_bfloat16 *w2;          // [n_expert][D][I]
};

struct Device {
    int device;               // the CUDA device
    int first_expert, n_expert;
    const __nv_bfloat16 *embed;       // [VOCAB][D]
    LayerWeights layers[N_LAYER];
    const __nv_bfloat16 *final_norm;  // [D]
    const __nv_bfloat16 *lm_head;     // [VOCAB][D]
};

// ---- the one function you write, in hw5.cu ------------------------------------------------
// Called once per case; the clock is running. main() has read the input and put the weights
// on the devices, and stops the clock when run() has returned and every device is synchronized.
//   devices     one per GPU, ordered by first_expert; their expert ranges tile 0 .. N_EXPERT-1
//   prompts     the N prompts, each a list of token ids in [0, VOCAB)
//   n_gen       n_gen[i] = K_i >= 1, the tokens to generate for prompt i;
//               prompts[i].size() + K_i - 1 <= N_CTX
//   tokens_out  tokens_out[i] has K_i entries: [0] the next token after prompt i, [t] the one
//               after feeding tokens_out[i][t-1]
//   hidden_out  [N][D]: the final-RMSNorm output at prompt i's last prompt position
// Both outputs are allocated and zeroed by main(), in host memory, and must be complete when
// run() returns. On entry device devices[0].device is current; run() may leave any current.
// main() may call run() again for another case with the same devices: everything run()
// allocated it frees before returning.
void run(const std::vector<Device> &devices, const std::vector<std::vector<int>> &prompts,
         const std::vector<int> &n_gen, std::vector<std::vector<int>> &tokens_out, std::vector<float> &hidden_out);

#endif
