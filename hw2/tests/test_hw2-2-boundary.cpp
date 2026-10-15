// In-memory checks isolate multiply() from the fixed driver's MPI file output.
#include "../hw2-2/hw2-2.cpp"

using Rows = std::vector<std::vector<Entry>>;

static Matrix shard(const Rows &rows, u32 cols, int rank, int ranks)
{
    Matrix m;
    m.rows = u32(rows.size());
    m.cols = cols;
    m.first = block_start(m.rows, rank, ranks);
    for (u32 i = m.first; i < block_start(m.rows, rank + 1, ranks); i++) {
        m.data.insert(m.data.end(), rows[i].begin(), rows[i].end());
        m.ptr.push_back(m.data.size());
    }
    return m;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        const HotConfig cfg = hot_config(MPI_COMM_WORLD, rank);
        if (!cfg.enabled || cfg.accum != AccumulatorMode::Auto || cfg.dense || cfg.emit_simd ||
            cfg.bytes != (u64(256) << 20))
            throw std::runtime_error("submission defaults are not hot=1, auto, dense=0, emit_simd=0, cache=256MiB");
        for (int test = 0; test < 6; test++) {
            const u32 M = test == 5 ? 1 : 3, K = 5, N = test == 5 ? 1 : 8192;
            Rows A(M), B(K);
            for (u32 i = 0; i < M; i++)
                for (u32 k = 0; k < K; k++)
                    if (test != 1 && (test >= 3 || k == 4)) A[i].push_back({k, 8});
            for (u32 k = 0; k < K; k++) {
                if (test == 2 || (test == 3 && k % 2)) continue;
                if (test == 4) {
                    for (u32 j = 0; j < N; j++) B[k].push_back({j, 1 + (j + k) % 8});
                } else B[k].push_back({N - 1, 8});
            }
            const Matrix a = shard(A, K, rank, ranks), b = shard(B, N, rank, ranks);
            for (int repeat = 0; repeat < 3; repeat++) {
                Matrix c;
                multiply(a, b, c);
                if (c.rows != M || c.cols != N || c.first != a.first || c.ptr.size() != a.ptr.size() ||
                    c.ptr.front() != 0 || c.ptr.back() != c.data.size())
                    throw std::runtime_error("boundary shape/offset mismatch");
                for (u32 il = 0; il < a.local_rows(); il++) {
                    std::vector<u32> expected(N, 0);
                    for (const Entry &x : A[a.first + il])
                        for (const Entry &y : B[x.col]) expected[y.col] += x.val * y.val;
                    u64 at = c.ptr[il];
                    for (u32 j = 0; j < N; j++) {
                        if (!expected[j]) continue;
                        if (at >= c.ptr[il + 1] || c.data[at].col != j || c.data[at].val != expected[j])
                            throw std::runtime_error("boundary value/canonical order mismatch");
                        at++;
                    }
                    if (at != c.ptr[il + 1]) throw std::runtime_error("unexpected boundary output entry");
                }
            }
        }
        if (!rank) std::printf("PASS defaults and 18 in-memory boundary checks at %d ranks\n", ranks);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
}
