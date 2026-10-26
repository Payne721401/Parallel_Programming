#pragma once
// Included inside hw3-1.cu's private namespace. Diagnostic scans never change grids.
struct EvolutionTile { unsigned alive, nonzero, changed, previous_nonzero; };
enum EvolutionField { EV_ALIVE, EV_NONZERO, EV_CHANGED, EV_NZ_TILES,
    EV_CHANGED_TILES, EV_INFLUENCE, EV_WORK, EV_EMPTY, EV_1_16, EV_17_128,
    EV_129_512, EV_513_UP, EV_FIELDS };
struct EvolutionRow {
    unsigned long long value[EV_FIELDS];
    int next_work, dense;
};

__global__ void evolution_scan(const uint8_t *current, const uint8_t *previous,
                               EvolutionTile *tiles, Problem p, int tiles_x) {
    const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int x0 = int(blockIdx.x) % tiles_x * TILE_W;
    const int y0 = int(blockIdx.x) / tiles_x * TILE_H;
    unsigned v[4]{};
    for (int cell = tid; cell < TILE_W*TILE_H; cell += 256) {
        const int x = x0 + cell % TILE_W, y = y0 + cell / TILE_W;
        if (x >= p.width || y >= p.height) continue;
        const size_t i = size_t(y)*p.width+x;
        const uint8_t now = current[i], old = previous ? previous[i] : 0;
        v[0] += now == 1;
        v[1] += now != 0;
        v[2] += previous && now != old; // All state bytes, including decay.
        v[3] += previous && old != 0;
    }
    __shared__ unsigned partial[4][8];
    for (int k = 0; k < 4; ++k) {
        for (int offset = 16; offset; offset /= 2)
            v[k] += __shfl_down_sync(0xffffffffu, v[k], offset);
        if (lane == 0) partial[k][warp] = v[k];
    }
    __syncthreads();
    if (warp == 0) {
        for (int k = 0; k < 4; ++k) {
            v[k] = lane < 8 ? partial[k][lane] : 0;
            for (int offset = 16; offset; offset /= 2)
                v[k] += __shfl_down_sync(0xffffffffu, v[k], offset);
        }
        if (lane == 0) tiles[blockIdx.x] = {v[0], v[1], v[2], v[3]};
    }
}

__global__ void evolution_reduce(const EvolutionTile *tiles, EvolutionRow *row,
                                  Problem p, int tiles_x, int count,
                                  const int *next_count, const int *dense) {
    const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int tile = int(blockIdx.x)*256+tid;
    unsigned long long v[EV_FIELDS]{};
    if (tile < count) {
        const auto state = tiles[tile];
        v[EV_ALIVE] = state.alive; v[EV_NONZERO] = state.nonzero;
        v[EV_CHANGED] = state.changed;
        v[EV_NZ_TILES] = state.nonzero != 0;
        v[EV_CHANGED_TILES] = state.changed != 0;
        const int bin = !state.nonzero ? EV_EMPTY : state.nonzero <= 16 ? EV_1_16
                      : state.nonzero <= 128 ? EV_17_128 : state.nonzero <= 512 ? EV_129_512 : EV_513_UP;
        v[bin] = 1;
        const bool torus = p.boundary == Boundary::Torus;
        const auto xs = tile_geometry::influence(tile % tiles_x, TILE_W, p.width, p.rule.radius, torus);
        const auto ys = tile_geometry::influence(tile / tiles_x, TILE_H, p.height, p.rule.radius, torus);
        bool needed = false;
        for (int y = 0; y < ys.count; ++y)
            for (int x = 0; x < xs.count; ++x)
                needed |= tiles[ys.ids[y]*tiles_x+xs.ids[x]].nonzero != 0;
        v[EV_INFLUENCE] = needed;
        v[EV_WORK] = needed || state.previous_nonzero != 0;
    }
    __shared__ unsigned long long partial[EV_FIELDS][8];
    for (int k = 0; k < EV_FIELDS; ++k) {
        for (int offset = 16; offset; offset /= 2)
            v[k] += __shfl_down_sync(0xffffffffu, v[k], offset);
        if (lane == 0) partial[k][warp] = v[k];
    }
    __syncthreads();
    if (warp == 0) {
        for (int k = 0; k < EV_FIELDS; ++k) {
            auto sum = lane < 8 ? partial[k][lane] : 0;
            for (int offset = 16; offset; offset /= 2)
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (lane == 0) atomicAdd(row->value+k, sum);
        }
    }
    if (blockIdx.x == 0 && tid == 0) {
        row->dense = dense ? *dense : 1;
        row->next_work = next_count ? (row->dense ? count : *next_count) : count;
    }
}

struct EvolutionProfile {
    bool enabled = false;
    int every = 128, used = 0, tiles_x = 0, tile_count = 0;
    EvolutionTile *tiles = nullptr;
    EvolutionRow *rows = nullptr;
    std::vector<int> generations;
    std::vector<cudaEvent_t> before, after;
    double setup_host_ms = 0;

    EvolutionProfile() {
        const char *flag = std::getenv("PP_EVOLVE");
        enabled = flag && std::strcmp(flag, "1") == 0;
        if (flag && !enabled && std::strcmp(flag, "0") != 0) {
            std::fprintf(stderr, "hw3-1: PP_EVOLVE must be 0 or 1\n");
            std::exit(EXIT_FAILURE);
        }
        if (!enabled) return;
        const char *interval = std::getenv("PP_EVOLVE_EVERY");
        if (interval) {
            char *end = nullptr;
            long value = std::strtol(interval, &end, 10);
            if (!*interval || *end || value < 1 || value > 5000) {
                std::fprintf(stderr, "hw3-1: PP_EVOLVE_EVERY must be 1..5000\n");
                std::exit(EXIT_FAILURE);
            }
            every = int(value);
        }
    }

    void prepare(const Problem &p) {
        if (!enabled) return;
        const auto start = PhaseProfile::Clock::now();
        for (int t = 0; t <= p.steps; ++t)
            if (t == 0 || t == p.steps || (t <= 64 && (t & (t-1)) == 0) || t % every == 0)
                generations.push_back(t);
        tiles_x = (p.width+TILE_W-1)/TILE_W;
        tile_count = tiles_x*((p.height+TILE_H-1)/TILE_H);
        check(cudaMalloc(&tiles, size_t(tile_count)*sizeof(EvolutionTile)), "allocate evolution tiles");
        check(cudaMalloc(&rows, generations.size()*sizeof(EvolutionRow)), "allocate evolution rows");
        before.resize(generations.size()); after.resize(generations.size());
        for (size_t i = 0; i < generations.size(); ++i) {
            check(cudaEventCreate(&before[i]), "create evolution event");
            check(cudaEventCreate(&after[i]), "create evolution event");
        }
        setup_host_ms = PhaseProfile::elapsed_ms(start);
    }

    void sample(int generation, const uint8_t *current, const uint8_t *previous,
                const Problem &p, const int *next_count, const int *dense) {
        if (!enabled || used == int(generations.size()) || generations[used] != generation) return;
        check(cudaEventRecord(before[used], 0), "begin evolution scan");
        check(cudaMemsetAsync(rows+used, 0, sizeof(EvolutionRow)), "clear evolution row");
        evolution_scan<<<tile_count, 256>>>(current, previous, tiles, p, tiles_x);
        check(cudaGetLastError(), "launch evolution scan");
        evolution_reduce<<<(tile_count+255)/256, 256>>>(tiles, rows+used, p, tiles_x, tile_count, next_count, dense);
        check(cudaGetLastError(), "launch evolution reduction");
        check(cudaEventRecord(after[used], 0), "end evolution scan");
        ++used;
    }

    void finish(const Problem &p, bool sparse) {
        if (!enabled) return;
        const auto collect_start = PhaseProfile::Clock::now();
        std::vector<EvolutionRow> host(used);
        check(cudaMemcpy(host.data(), rows, size_t(used)*sizeof(EvolutionRow), cudaMemcpyDeviceToHost), "read evolution rows");
        const double collect_ms = PhaseProfile::elapsed_ms(collect_start);
        std::vector<float> scan_ms(used), update_ms(used, 0);
        double total_scan = 0;
        int fixed_at = -1;
        for (int i = 0; i < used; ++i) {
            check(cudaEventElapsedTime(&scan_ms[i], before[i], after[i]), "time evolution scan");
            if (i) check(cudaEventElapsedTime(&update_ms[i], after[i-1], before[i]), "time evolution update interval");
            total_scan += scan_ms[i];
            if (generations[i] > 0 && !host[i].value[EV_CHANGED] && fixed_at < 0) fixed_at = generations[i];
        }
        const auto free_start = PhaseProfile::Clock::now();
        for (auto event : before) check(cudaEventDestroy(event), "destroy evolution event");
        for (auto event : after) check(cudaEventDestroy(event), "destroy evolution event");
        check(cudaFree(rows), "free evolution rows");
        check(cudaFree(tiles), "free evolution tiles");
        const double free_ms = PhaseProfile::elapsed_ms(free_start);
        for (int i = 0; i < used; ++i) {
            const auto &v = host[i].value;
            std::fprintf(stderr,
                "[PP_EVOLVE] generation=%d alive=%llu nonzero=%llu decay=%llu changed=%llu "
                "nonzero_tiles=%llu changed_tiles=%llu influence_tiles=%llu work_tiles=%llu "
                "empty_tiles=%llu tiles_1_16=%llu tiles_17_128=%llu tiles_129_512=%llu tiles_513_up=%llu "
                "scheduled_next=%d dense=%d fixed=%d scan_ms=%.6f update_ms=%.6f\n",
                generations[i], v[EV_ALIVE], v[EV_NONZERO], v[EV_NONZERO]-v[EV_ALIVE], v[EV_CHANGED],
                v[EV_NZ_TILES], v[EV_CHANGED_TILES], v[EV_INFLUENCE], v[EV_WORK],
                v[EV_EMPTY], v[EV_1_16], v[EV_17_128], v[EV_129_512], v[EV_513_UP],
                host[i].next_work, host[i].dense, int(generations[i] > 0 && !v[EV_CHANGED]),
                double(scan_ms[i]), double(update_ms[i]));
        }
        std::fprintf(stderr,
            "[PP_EVOLVE_META] steps=%d cells=%zu tiles=%d samples=%d every=%d sparse=%d "
            "first_observed_fixed=%d setup_host_ms=%.6f scans_ms=%.6f collect_host_ms=%.6f free_host_ms=%.6f\n",
            p.steps, size_t(p.width)*p.height, tile_count, used, every, int(sparse), fixed_at,
            setup_host_ms, total_scan, collect_ms, free_ms);
    }
};
