// HW 5: main. FIXED FILE: the judge builds with its own copy, so editing yours changes nothing
// there. It reads the weight file onto the GPUs in the layout mixtral.hpp describes, then for
// each case on the command line reads the input, calls your run() in hw5.cu, writes the output
// and prints the time. Read it to see what run() is given and what it must leave behind.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mixtral.hpp"

constexpr int MAX_DEVICES = 2;   // the evaluation's two GPUs; more are ignored, fewer cannot hold the model

[[noreturn]] static void die(const std::string &msg)
{
    std::cerr << "hw5: " << msg << "\n";
    std::exit(1);
}

static void cuda_check(cudaError_t e, const char *what)
{
    if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}

// ---- the input --------------------------------------------------------------------------------
// Text:  N, then N lines  L K tok_1 ... tok_L  (K = tokens to generate for the prompt).
static std::vector<std::vector<int>> read_prompts(const std::string &path, std::vector<int> &n_gen)
{
    std::ifstream f(path);
    if (!f) die("failed to open input file " + path);
    int n = -1;
    f >> n;
    if (!f || n < 0) die("malformed input file " + path);
    std::vector<std::vector<int>> prompts(n);
    n_gen.assign(n, 0);
    for (int i = 0; i < n; i++) {
        int L = 0;
        f >> L >> n_gen[i];
        if (!f || L < 1 || n_gen[i] < 1 || L + n_gen[i] - 1 > N_CTX) die("bad prompt length or K in " + path);
        prompts[i].resize(L);
        for (int &tok : prompts[i]) {
            f >> tok;
            if (!f || tok < 0 || tok >= VOCAB) die("bad token id in " + path);
        }
    }
    return prompts;
}

// ---- the weight file: MIXTRAL1 ----------------------------------------------------------------
// "MIXTRAL1", int32[16] (n_layer, n_expert, top_k, d, I, n_head, n_kv, head_dim, vocab, zeros),
// then the tensors below in order, each starting at a multiple of 4096 bytes, every one bf16:
//
//   embed       [VOCAB][D]
//   per layer:  attn_norm [D]; qkv_w [QKV_DIM][D]; o_w [D][Q_DIM]; mlp_norm [D];
//               router_w [N_EXPERT][D]; w1 [N_EXPERT][2I][D]; w2 [N_EXPERT][D][I]
//   final_norm  [D]
//   lm_head     [VOCAB][D]

constexpr size_t FILE_ALIGN = 4096;
constexpr int32_t EXPECTED_HEADER[9] = {N_LAYER, N_EXPERT, TOP_K, D, I, N_HEAD, N_KV, HEAD_DIM, VOCAB};

// The file, read tensor by tensor into one pinned buffer.
struct Reader {
    std::ifstream f;
    size_t pos = 0;
    void *hbuf = nullptr;
    size_t hbuf_bytes = 0;

    explicit Reader(const std::string &path) : f(path, std::ios::binary)
    {
        if (!f) die("failed to open " + path);
        char magic[8];
        int32_t hdr[16];
        f.read(magic, 8);
        f.read(reinterpret_cast<char *>(hdr), sizeof hdr);
        if (!f || std::memcmp(magic, "MIXTRAL1", 8) != 0) die("not a MIXTRAL1 weight file: " + path);
        for (int i = 0; i < 9; i++)
            if (hdr[i] != EXPECTED_HEADER[i]) die("the weight file is not Mixtral-8x7B's (header field " + std::to_string(i) + ")");
        pos = 8 + sizeof hdr;
    }
    // The next tensor's `bytes` into hbuf.
    void next(size_t bytes)
    {
        const size_t aligned = (pos + FILE_ALIGN - 1) / FILE_ALIGN * FILE_ALIGN;
        f.seekg(std::streamoff(aligned));
        if (bytes > hbuf_bytes) {
            if (hbuf) cuda_check(cudaFreeHost(hbuf), "cudaFreeHost");
            cuda_check(cudaMallocHost(&hbuf, bytes), "cudaMallocHost for the weights");
            hbuf_bytes = bytes;
        }
        f.read(static_cast<char *>(hbuf), std::streamsize(bytes));
        if (!f) die("short read on the weight file");
        pos = aligned + bytes;
    }
};

// Everything one device holds: one arena, carved in the order of Device's fields.
struct Arena {
    int device;
    char *base = nullptr;
    size_t size = 0, used = 0;
    __nv_bfloat16 *take(size_t n)
    {
        used = (used + 255) / 256 * 256;
        __nv_bfloat16 *p = reinterpret_cast<__nv_bfloat16 *>(base + used);
        used += n * sizeof(__nv_bfloat16);
        if (used > size) die("arena overflow");   // cannot happen: sized by the same walk
        return p;
    }
};

static size_t device_bytes(int n_expert)
{
    size_t shared = size_t(VOCAB) * D * 2 + D;   // embed, lm_head, final_norm
    size_t layer = D + size_t(QKV_DIM) * D + size_t(D) * Q_DIM + D + size_t(N_EXPERT) * D;
    size_t experts = size_t(n_expert) * (size_t(2 * I) * D + size_t(D) * I);
    return (shared + N_LAYER * (layer + experts)) * sizeof(__nv_bfloat16) + N_LAYER * 8 * 256;   // plus alignment slack
}

// One tensor of n values, read and copied to every device.
static void put_shared(Reader &r, size_t n, const std::vector<Arena> &arenas, std::vector<__nv_bfloat16 *> dst)
{
    r.next(n * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < arenas.size(); i++) {
        cuda_check(cudaSetDevice(arenas[i].device), "cudaSetDevice");
        cuda_check(cudaMemcpy(dst[i], r.hbuf, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "uploading a tensor");
    }
}

// An expert tensor of N_EXPERT experts of `per_expert` values each: each device takes its slice.
static void put_experts(Reader &r, size_t per_expert, const std::vector<Device> &devices, std::vector<__nv_bfloat16 *> dst)
{
    r.next(size_t(N_EXPERT) * per_expert * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < devices.size(); i++) {
        const Device &dv = devices[i];
        cuda_check(cudaSetDevice(dv.device), "cudaSetDevice");
        cuda_check(cudaMemcpy(dst[i], static_cast<const char *>(r.hbuf) + size_t(dv.first_expert) * per_expert * sizeof(__nv_bfloat16),
                              size_t(dv.n_expert) * per_expert * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "uploading the experts");
    }
}

static size_t free_memory(int device)
{
    size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    return free_bytes;
}

int main(int argc, char *argv[])
{
    if (argc < 4 || argc % 2 != 0) {
        std::cerr << "Usage: ./hw5 <weights.bin> <input.txt> <output.bin> [<input.txt> <output.bin> ...]\n";
        return 1;
    }
    const std::string weight_path = argv[1];

    /////////////////////////////////////////////////////////////
    // Outside the clock: the CUDA contexts and the weights on the devices.
    auto l0 = std::chrono::steady_clock::now();
    int n_dev = 0;
    cuda_check(cudaGetDeviceCount(&n_dev), "cudaGetDeviceCount");
    if (n_dev < 1) die("no CUDA device");
    n_dev = std::min(n_dev, MAX_DEVICES);
    std::vector<Device> devices(n_dev);
    std::vector<Arena> arenas(n_dev);
    for (int i = 0; i < n_dev; i++) {
        Device &dv = devices[i];
        dv.device = i;
        dv.first_expert = N_EXPERT * i / n_dev;
        dv.n_expert = N_EXPERT * (i + 1) / n_dev - dv.first_expert;
        cuda_check(cudaSetDevice(i), "cudaSetDevice");
        cuda_check(cudaFree(nullptr), "creating the CUDA context");
        Arena &a = arenas[i];
        a.device = i;
        a.size = device_bytes(dv.n_expert);
        const size_t free_bytes = free_memory(i);
        if (a.size > free_bytes)
            die("device " + std::to_string(i) + " has " + std::to_string(free_bytes >> 30) + " GB free, the model's " +
                std::to_string(dv.n_expert) + " experts of every layer need " + std::to_string(a.size >> 30) + " GB" +
                (n_dev < MAX_DEVICES ? ": the model needs two GPUs" : ""));
        cuda_check(cudaMalloc(&a.base, a.size), "cudaMalloc for the weights");
        // Carve the arena in the file's order, so the fields are laid out as the file is read.
        dv.embed = a.take(size_t(VOCAB) * D);
        for (int l = 0; l < N_LAYER; l++) {
            LayerWeights &L = dv.layers[l];
            L.attn_norm = a.take(D);
            L.qkv_w = a.take(size_t(QKV_DIM) * D);
            L.o_w = a.take(size_t(D) * Q_DIM);
            L.mlp_norm = a.take(D);
            L.router_w = a.take(size_t(N_EXPERT) * D);
            L.w1 = a.take(size_t(dv.n_expert) * 2 * I * D);
            L.w2 = a.take(size_t(dv.n_expert) * D * I);
        }
        dv.final_norm = a.take(D);
        dv.lm_head = a.take(size_t(VOCAB) * D);
    }

    {
        Reader r(weight_path);
        auto all = [&](auto field) {
            std::vector<__nv_bfloat16 *> v;
            for (Device &dv : devices) v.push_back(const_cast<__nv_bfloat16 *>(field(dv)));
            return v;
        };
        put_shared(r, size_t(VOCAB) * D, arenas, all([](Device &dv) { return dv.embed; }));
        for (int l = 0; l < N_LAYER; l++) {
            put_shared(r, D, arenas, all([l](Device &dv) { return dv.layers[l].attn_norm; }));
            put_shared(r, size_t(QKV_DIM) * D, arenas, all([l](Device &dv) { return dv.layers[l].qkv_w; }));
            put_shared(r, size_t(D) * Q_DIM, arenas, all([l](Device &dv) { return dv.layers[l].o_w; }));
            put_shared(r, D, arenas, all([l](Device &dv) { return dv.layers[l].mlp_norm; }));
            put_shared(r, size_t(N_EXPERT) * D, arenas, all([l](Device &dv) { return dv.layers[l].router_w; }));
            put_experts(r, size_t(2 * I) * D, devices, all([l](Device &dv) { return dv.layers[l].w1; }));
            put_experts(r, size_t(D) * I, devices, all([l](Device &dv) { return dv.layers[l].w2; }));
        }
        put_shared(r, D, arenas, all([](Device &dv) { return dv.final_norm; }));
        put_shared(r, size_t(VOCAB) * D, arenas, all([](Device &dv) { return dv.lm_head; }));
        if (r.hbuf) cuda_check(cudaFreeHost(r.hbuf), "cudaFreeHost");
    }
    for (int i = 0; i < n_dev; i++) {
        cuda_check(cudaSetDevice(i), "cudaSetDevice");
        cuda_check(cudaDeviceSynchronize(), "uploading the weights");
    }
    std::vector<size_t> free_before(n_dev);
    for (int i = 0; i < n_dev; i++) free_before[i] = free_memory(i);
    auto l1 = std::chrono::steady_clock::now();
    std::cout << "Devices: " << n_dev << ", experts per device: " << devices[0].n_expert << ", weights per device: "
              << (arenas[0].used >> 20) / 1024.0 << " GB, free: " << (free_before[0] >> 20) / 1024.0 << " GB\n";
    std::cout << "Load time: " << std::chrono::duration<double>(l1 - l0).count() << " s (CUDA contexts and weights; not counted)" << std::endl;
    /////////////////////////////////////////////////////////////

    for (int c = 2; c + 1 < argc; c += 2) {
        const std::string input_path = argv[c], output_path = argv[c + 1];
        std::vector<int> n_gen;
        std::vector<std::vector<int>> prompts = read_prompts(input_path, n_gen);
        std::vector<std::vector<int>> tokens(prompts.size());
        for (size_t i = 0; i < prompts.size(); i++) tokens[i].assign(size_t(n_gen[i]), 0);
        std::vector<float> hidden(prompts.size() * D, 0.f);

        /////////////////////////////////////////////////////////////
        // Timed section. The clock: run() and nothing else, until every device is idle.
        cuda_check(cudaSetDevice(devices[0].device), "cudaSetDevice");
        auto t0 = std::chrono::steady_clock::now();
        run(devices, prompts, n_gen, tokens, hidden);
        for (int i = 0; i < n_dev; i++) {
            cuda_check(cudaSetDevice(i), "cudaSetDevice");
            cuda_check(cudaDeviceSynchronize(), "after run()");
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_dev; i++) {
            cuda_check(cudaSetDevice(i), "cudaSetDevice");
            cuda_check(cudaGetLastError(), "after run()");
        }
        /////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////
        // Output section. The judge parses this file.
        //   int32 N
        //   N times: int32 L, int32 K, int32 tokens[K], float32 hidden[D]
        for (size_t i = 0; i < prompts.size(); i++)
            if (tokens[i].size() != size_t(n_gen[i])) die("run() changed the size of tokens_out[" + std::to_string(i) + "]");
        std::ofstream ofs(output_path, std::ios::binary);
        if (!ofs) die("failed to open " + output_path + " for writing");
        int32_t n = int32_t(prompts.size());
        ofs.write(reinterpret_cast<const char *>(&n), sizeof n);
        for (size_t i = 0; i < prompts.size(); i++) {
            int32_t L = int32_t(prompts[i].size()), K = n_gen[i];
            ofs.write(reinterpret_cast<const char *>(&L), sizeof L);
            ofs.write(reinterpret_cast<const char *>(&K), sizeof K);
            ofs.write(reinterpret_cast<const char *>(tokens[i].data()), sizeof(int32_t) * K);
            ofs.write(reinterpret_cast<const char *>(hidden.data() + i * D), sizeof(float) * D);
        }
        ofs.close();
        if (!ofs) die("failed to write " + output_path);
        /////////////////////////////////////////////////////////////

        size_t total_tokens = 0, total_gen = 0;
        for (auto &p : prompts) total_tokens += p.size();
        for (int k : n_gen) total_gen += k;
        std::cout << "Case " << input_path << ": prompts " << prompts.size() << ", tokens " << total_tokens << ", generate "
                  << total_gen << "\n";
        // Flushed at once: a run stopped at its time limit loses what the buffer held, and the
        // judge times each case it finished by this line.
        std::cout << "Compute time: " << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;
        for (int i = 0; i < n_dev; i++) {
            const size_t now = free_memory(i);
            if (now + (size_t(256) << 20) < free_before[i])
                std::cout << "Warning: device " << i << " has " << ((free_before[i] - now) >> 20)
                          << " MB less free memory than before run(): run() leaks\n";
        }
    }

    for (int i = 0; i < n_dev; i++) {
        cuda_check(cudaSetDevice(i), "cudaSetDevice");
        cuda_check(cudaFree(arenas[i].base), "cudaFree");
    }
    return 0;
}
