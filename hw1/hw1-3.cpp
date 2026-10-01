/*
HW 1-3: heat diffusion through a heterogeneous block with reactive inclusions.

    hw1-3 <N> <T> <seed> <theta> <output>

An N x N x N block of material whose conductivity varies from cell to cell
starts at a random temperature and is left to diffuse for up to T time steps,
or until its energy has fallen to theta times what it started with. Both the
temperature field u and the conductivity field a are generated from <seed>,
and so are the inclusions: up to four spheres of a reactive material that
absorbs heat, the faster the hotter it is. The answer is the number of steps
taken, the energy left, and a 32 x 32 x 32 lattice of sample temperatures,
written to <output>.

One step moves heat across each of a cell's six faces at the rate of the two
cells' mean conductivity:

    r[p] = u[p] + (1/12) * sum over the 6 neighbors q of (a[p] + a[q]) * (u[q] - u[p])

and for a plain cell that is the new temperature, u'[p] = r[p]. A reactive
cell also loses heat to its reaction during the step, at a rate that grows with
its own new temperature, so the new temperature is the root of

    u'[p] + R * (exp(u'[p]) - 1) = r[p]

which the program takes NEWTON iterations of Newton's method from r[p] to reach.
Cells outside the block (the halo) have u = 0 and a = 0. Conductivities lie in
[0, 1), so the weight on u[p] never goes negative and the scheme is stable at
any N and T. The energy of the block is the sum of u[p]^2 over its cells; the
program computes it after every step and stops once it is theta times the
initial energy or less.
*/

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include <chrono>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>

// Transparent huge pages: a 7 GB array needs 1.7M 4 KB faults to materialise
// and 1.7M TLB entries to address, against a TLB holding one or two thousand.
// 2 MB pages cut both by 512. glibc hands back a pointer just past a chunk
// header, so round up to a page boundary first.
static void adviseHuge(void* p, size_t bytes) {
#ifdef MADV_HUGEPAGE
    const uintptr_t pg = 4096;
    const uintptr_t start = (uintptr_t)p;
    const uintptr_t aligned = (start + pg - 1) & ~(pg - 1);
    const size_t skip = (size_t)(aligned - start);
    if (bytes > skip) madvise((void*)aligned, bytes - skip, MADV_HUGEPAGE);
#else
    (void)p;
    (void)bytes;
#endif
}

// -DSTENCIL_SIMD=0 drops the omp simd on the diffusion loop and lets GCC's
// cost model decide instead.
#ifndef STENCIL_SIMD
#define STENCIL_SIMD 1
#endif

// Per-thread profile counters, one 64-byte line each so the threads never
// share a line. Slots: pass 1 work, wait after pass 1, pass 2 work, wait after 2.
#define TPROF_STRIDE 8
#define TPROF_SLOTS 4

// ---------- staged timing ----------
// PP_TIMING=1 srun -n 1 -c 8 ./hw1-3 768 6 1087 0 out.txt
static inline double nowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Spec 1.4: "Use sched_getaffinity and CPU_COUNT to determine how many cores
// are available to your program, and create that many threads." The cluster
// overwrites OMP_NUM_THREADS to 1, so the mask is the only reliable source.
static int usableCpus() {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
    return omp_get_max_threads();
}

// Outside the protected range as of the 2026-09-29 handout, but the reference
// answer is defined by this exact iteration count rather than by the converged
// root, so changing the values still changes the answer.
static const double R = 0.5;    // the reaction's strength
static const int SUBSTEPS = 4;  // sub-steps of the reaction per time step
static const int NEWTON = 3;    // Newton iterations per sub-step

// ========== START: DO NOT CHANGE BELOW ==========
static double field(uint64_t seed, uint64_t index, int which) {
    uint64_t x = (index * 2 + which) ^ (seed * 0x9E3779B97F4A7C15ull);
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (x >> 11) * (1.0 / 9007199254740992.0);
}

// The inclusions: seed % 5 spheres, each a center in cell coordinates (1..N
// along each axis) and a squared radius, drawn from field() past the cells'
// own indices. Returns how many there are.
struct Inclusion {
    double ci, cj, ck, r2;
};
static int inclusions(uint64_t seed, long N, Inclusion* out) {
    const int B = (int)(seed % 5);
    const uint64_t base = (uint64_t)N * N * N;
    for (int b = 0; b < B; b++) {
        out[b].ci = 1 + (0.2 + 0.6 * field(seed, base + 2 * b, 0)) * (N - 1);
        out[b].cj = 1 + (0.2 + 0.6 * field(seed, base + 2 * b, 1)) * (N - 1);
        out[b].ck = 1 + (0.2 + 0.6 * field(seed, base + 2 * b + 1, 0)) * (N - 1);
        const double r = (0.12 + 0.06 * field(seed, base + 2 * b + 1, 1)) * N;
        out[b].r2 = r * r;
    }
    return B;
}
// =========== END: DO NOT CHANGE ABOVE ===========

// A maximal run of consecutive reactive cells along k, as a flat index and a
// length. Inclusions are spheres, so for a given (i, j) each contributes at
// most one contiguous range of k, and four of them at most four disjoint runs.
// About 3 MB at N=960 against 890 MB for a full per-cell mat array, and one
// fewer read stream in every time step.
struct Run {
    long p;
    int len;
};

// Whether cell (i, j, k), 1-based, lies inside any of the B inclusions.
static bool reactive(const Inclusion* inc, int B, long i, long j, long k) {
    for (int b = 0; b < B; b++) {
        const double di = i - inc[b].ci, dj = j - inc[b].cj, dk = k - inc[b].ck;
        if (di * di + dj * dj + dk * dk <= inc[b].r2) return true;
    }
    return false;
}

// The new temperature of a reactive cell whose plain update would be r:
// NEWTON iterations of Newton's method on x + R (e^x - 1) = r, from x = r.
//
// Advances RBLK cells at once. One cell's chain is SUBSTEPS * NEWTON = 12
// exp() calls deep and each one feeds the next, so a lone cell spends nearly
// all of its time waiting on latency with the execution units idle. Cells are
// independent of each other, so interleaving them lets the out-of-order engine
// overlap one cell's exp() with the next cell's and the chain depth stops being
// the limit. Every cell still performs the same operations in the same order,
// so each result is bit-identical to the one-at-a-time version.
//
// Not omp simd: a vector exp() means libmvec, which GCC only emits under
// -ffast-math and which does not round the same way as the scalar one.
#ifndef RBLK
#define RBLK 8
#endif
static void reactBlock(double* x, int n) {
    double prev[RBLK];
    for (int s = 0; s < SUBSTEPS; s++) {
        for (int c = 0; c < n; c++) prev[c] = x[c];
        for (int it = 0; it < NEWTON; it++)
            for (int c = 0; c < n; c++) {
                const double ex = exp(x[c]);
                x[c] -= (x[c] + R * (ex - 1.0) - prev[c]) / (1.0 + R * ex);
            }
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <N> <T> <seed> <theta> <output>\n", argv[0]);
        return 1;
    }
    const long N = atol(argv[1]);
    const int T = atoi(argv[2]);
    const uint64_t seed = strtoull(argv[3], nullptr, 10);
    const double theta = atof(argv[4]);
    if (N < 32 || T < 0 || theta < 0) {
        fprintf(stderr, "N must be at least 32, T and theta non-negative\n");
        return 1;
    }

    int nthreads = usableCpus();
    if (const char* e = getenv("PP_THREADS")) {   // our own knob, for sweeps
        int v = atoi(e);
        if (v > 0) nthreads = v;
    }
    omp_set_num_threads(nthreads);

    const double tStart = nowMs();

    // (N+2)^3 with a halo of zeros around the block, so no cell is a special case.
    const long M = N + 2;
    const long SI = M * M, SJ = M;  // strides of i and j; k is contiguous
    // calloc, not std::vector: a vector value-initialises, memsetting 11 GB of
    // zeros at N=768 on one thread before a single cell is computed. calloc at
    // this size gets kernel anonymous pages that are already zero, so the pages
    // materialise on first touch -- in the parallel loop below, which also puts
    // each page on the NUMA node of the thread that will keep using it.
    const size_t cells = (size_t)M * M * M;
    // Two O(N^3) arrays now, u and a. unew and the per-cell reactive byte are
    // both gone: 14.2 GB at N=960 against 22.3 GB and a 16 GB limit. Keep
    // BYTES_PER_CELL in step -- both PP_TIMING and the OOM message use it.
    const double BYTES_PER_CELL = 16.0;
    double* u = (double*)calloc(cells, sizeof(double));
    double* a = (double*)calloc(cells, sizeof(double));
    if (!u || !a) {
        fprintf(stderr, "out of memory for N=%ld (%.2f GB)\n",
                N, cells * BYTES_PER_CELL / (1024.0 * 1024.0 * 1024.0));
        return 1;
    }
    adviseHuge(u, cells * sizeof(double));
    adviseHuge(a, cells * sizeof(double));

    const double tAlloc = nowMs();

    Inclusion inc[4];
    const int B = inclusions(seed, N, inc);

    // field() is a pure function of the index, so every cell is independent --
    // no sequential PRNG state to carry. Running it in parallel also
    // first-touches each page from the thread that will keep using it.
    #pragma omp parallel for schedule(static)
    for (long i = 1; i <= N; i++)
        for (long j = 1; j <= N; j++) {
            const long base = i * SI + j * SJ;
            const uint64_t idxBase = (uint64_t)((i - 1) * N + (j - 1)) * N;
            for (long k = 1; k <= N; k++) {
                u[base + k] = field(seed, idxBase + (uint64_t)(k - 1), 0);
                a[base + k] = field(seed, idxBase + (uint64_t)(k - 1), 1);
            }
        }

    // The reactive set, as runs of consecutive k. A cell is inside a sphere
    // when dk^2 <= r^2 - di^2 - dj^2, which for a given (i, j) is one interval
    // of k bounded by ck +- sqrt(r^2 - di^2 - dj^2). Widening those bounds by a
    // cell gives a window that certainly contains every reactive k, and
    // reactive() then decides each k inside it -- so the set is exactly what a
    // full scan produces, with the predicate evaluated O(reactive cells) times
    // instead of N^3.
    //
    // Each thread owns whole i planes, so the per-plane buffers need no lock,
    // and concatenating them in i order keeps the list deterministic.
    std::vector<std::vector<Run> > runsPerPlane(N + 1);

    if (B > 0) {
        #pragma omp parallel for schedule(static)
        for (long i = 1; i <= N; i++) {
            std::vector<Run>& out = runsPerPlane[i];
            for (long j = 1; j <= N; j++) {
                long wLo = N + 1, wHi = 0;
                for (int b = 0; b < B; b++) {
                    const double di = (double)i - inc[b].ci;
                    const double dj = (double)j - inc[b].cj;
                    const double rest = inc[b].r2 - di * di - dj * dj;
                    if (rest < 0.0) continue;
                    const double half = std::sqrt(rest);
                    long lo = (long)std::floor(inc[b].ck - half) - 1;
                    long hi = (long)std::ceil(inc[b].ck + half) + 1;
                    if (lo < 1) lo = 1;
                    if (hi > N) hi = N;
                    if (lo < wLo) wLo = lo;
                    if (hi > wHi) wHi = hi;
                }
                if (wLo > wHi) continue;   // this (i, j) misses every inclusion

                long runStart = -1;
                for (long k = wLo; k <= wHi; k++) {
                    const long p = i * SI + j * SJ + k;
                    if (reactive(inc, B, i, j, k)) {
                        if (runStart < 0) runStart = p;
                    } else if (runStart >= 0) {
                        out.push_back(Run{runStart, (int)(p - runStart)});
                        runStart = -1;
                    }
                }
                if (runStart >= 0) {
                    const long pEnd = i * SI + j * SJ + (wHi + 1);
                    out.push_back(Run{runStart, (int)(pEnd - runStart)});
                }
            }
        }
    }

    std::vector<Run> runs;
    for (long i = 1; i <= N; i++)
        runs.insert(runs.end(), runsPerPlane[i].begin(), runsPerPlane[i].end());
    std::vector<std::vector<Run> >().swap(runsPerPlane);   // give the memory back
    const long nRuns = (long)runs.size();

    const double tGen = nowMs();

    // The energy of the block: the sum of u^2 over its cells. Only energy0
    // uses this now -- the per-step energy is fused into the stencil loop, so
    // the array is swept once per step instead of twice.
    auto energy_of = [&](const double* v) {
        double energy = 0.0;
        #pragma omp parallel for schedule(static) reduction(+ : energy)
        for (long i = 1; i <= N; i++)
            for (long j = 1; j <= N; j++)
                for (long k = 1; k <= N; k++) {
                    const double x = v[i * SI + j * SJ + k];
                    energy += x * x;
                }
        return energy;
    };

    const double energy0 = energy_of(u);
    double energy = energy0;
    int steps = 0;

    const double tEnergy0 = nowMs();
    double tStencilAcc = 0.0;   // the per-step energy is folded into this now

    // Totals hide imbalance: eight threads averaging 100 ms say nothing about
    // one of them taking 400 ms while the rest waited. Time each pass and each
    // barrier per thread instead, and report max against mean.
    //
    // Gated on prof, because the clock reads and the barrier that separates the
    // wait from the work are per step and per thread, and the judge never asks
    // for them. prof is read once here, so it is uniform across the team --
    // which is what makes it legal to put a barrier under it.
    const bool prof = (getenv("PP_TIMING") != nullptr);
    std::vector<double> tprof((size_t)nthreads * TPROF_STRIDE, 0.0);

    // Scratch for the in-place update. Writing the new u[i] destroys values
    // that plane i+1, row j+1 and cell k+1 still need, so each thread carries
    // the old plane i-1 and the old rows j-1 and j, and reads the three "+1"
    // neighbours out of u while they are still untouched. firstPlanes is how a
    // thread publishes its own first plane to the thread below, whose last
    // plane needs it. O(N^2): 118 MB at N=960 against the 7.1 GB unew cost.
    const size_t planeDoubles = (size_t)M * M;
    double* planeBufs = (double*)calloc((size_t)nthreads * planeDoubles, sizeof(double));
    double* firstPlanes = (double*)calloc((size_t)nthreads * planeDoubles, sizeof(double));
    double* rowBufs = (double*)calloc((size_t)nthreads * 2 * M, sizeof(double));
    if (!planeBufs || !firstPlanes || !rowBufs) {
        fprintf(stderr, "out of memory for the in-place scratch buffers\n");
        return 1;
    }
    adviseHuge(planeBufs, (size_t)nthreads * planeDoubles * sizeof(double));
    adviseHuge(firstPlanes, (size_t)nthreads * planeDoubles * sizeof(double));

    while (steps < T) {
        const double s0 = nowMs();

        double e = 0.0;

        // One parallel region for both passes.
        #pragma omp parallel reduction(+ : e)
        {
            const int tid = omp_get_thread_num();
            const int P = omp_get_num_threads();
            double* tp = &tprof[(size_t)tid * TPROF_STRIDE];

            // A manual slab split rather than omp for: the rolling buffers
            // have to know which planes this thread owns, and which plane the
            // thread above is about to overwrite. Work per plane is uniform, so
            // spread the remainder one plane at a time rather than rounding the
            // chunk up -- at N=700 with 8 threads, ceil(700/8) = 89 would give
            // seven threads 89 planes and the last 77, a 13% gap the whole team
            // waits on. This gives four threads 88 and four 87.
            const long span = N / P, extra = N % P;
            const long cnt = span + (tid < extra ? 1 : 0);
            const long lo = 1 + (long)tid * span + (tid < extra ? tid : extra);
            const long hi = lo + cnt - 1;
            const bool mine = (cnt > 0);

            double* plane = planeBufs + (size_t)tid * planeDoubles;
            double* myFirst = firstPlanes + (size_t)tid * planeDoubles;
            double* row = rowBufs + (size_t)tid * 2 * M;
            double* cur = row + M;

            const double w0 = prof ? omp_get_wtime() : 0.0;

            // Seed the rolling plane from the old plane below this slab and
            // publish this slab's own first plane, both before anyone starts
            // overwriting. Planes 0 and N+1 are halo and never written, so the
            // outermost threads need nothing from a neighbour.
            if (mine) {
                memcpy(plane, u + (lo - 1) * SI, planeDoubles * sizeof(double));
                memcpy(myFirst, u + lo * SI, planeDoubles * sizeof(double));
            }
            #pragma omp barrier

            // Pass 1: plain diffusion for every cell, in place. With the
            // reaction moved to pass 2 the k loop has no branch and no call,
            // and everything it reads comes from a buffer or from part of u
            // this thread has not reached, so it carries no dependence.
            //
            // omp simd is what makes e += r*r legal to vectorise -- GCC will
            // not regroup the additions unprompted without -ffast-math, which
            // we cannot afford globally. It also spares the runtime aliasing
            // check around the u[p] write and the u[p + SJ] read. Temperatures
            // are unaffected: every u[p] is the same expression, same order.
            if (mine) {
                for (long i = lo; i <= hi; i++) {
                    // Old plane i+1: untouched inside the slab, but at the top
                    // the thread above overwrites its first plane immediately,
                    // so read the copy it published.
                    const double* up1 =
                        (i < hi) ? (u + (i + 1) * SI)
                                 : ((tid + 1 < P && hi < N)
                                        ? firstPlanes + (size_t)(tid + 1) * planeDoubles
                                        : u + (hi + 1) * SI);

                    // Row 0 of this plane is halo, so it is still the old row.
                    memcpy(row, u + i * SI, M * sizeof(double));

                    for (long j = 1; j <= N; j++) {
                        const long base = i * SI + j * SJ;
                        const long qb = j * SJ;
                        // Stash the old row before overwriting any of it, so
                        // the k-1 and k+1 neighbours stay available.
                        memcpy(cur, u + base, M * sizeof(double));
#if STENCIL_SIMD
                        #pragma omp simd reduction(+ : e)
#endif
                        for (long k = 1; k <= N; k++) {
                            const long p = base + k;
                            const long q = qb + k;
                            const double up = cur[k], ap = a[p];
                            double flux = 0.0;
                            flux += (ap + a[p - SI]) * (plane[q] - up);
                            flux += (ap + a[p + SI]) * (up1[q] - up);
                            flux += (ap + a[p - SJ]) * (row[k] - up);
                            flux += (ap + a[p + SJ]) * (u[p + SJ] - up);
                            flux += (ap + a[p - 1]) * (cur[k - 1] - up);
                            flux += (ap + a[p + 1]) * (cur[k + 1] - up);
                            const double r = up + flux * (1.0 / 12.0);
                            plane[q] = up;   // this plane becomes "i-1" next
                            u[p] = r;
                            e += r * r;
                        }
                        std::swap(row, cur);   // this row becomes "j-1" next
                    }
                }
            }

            const double w1 = prof ? omp_get_wtime() : 0.0;
            double w2 = w1, w3 = w1;

            // Pass 2: replace the plain value in each reactive cell with the
            // reacted one. Pass 1 already added pre^2 to the energy, so adding
            // the difference costs O(reactive cells) instead of another full
            // sweep. dynamic: a run is a chord through a sphere, so lengths
            // vary from 1 to twice the radius.
            //
            // Skipped outright when the seed yields no inclusions -- four of
            // the ten public cases. nRuns is shared and fixed before the region
            // starts, so every thread takes the same branch, which is what a
            // barrier and an omp for require: the whole team, or none of it.
            if (nRuns > 0) {
                // A run can lie in any slab, so pass 2 reads cells that other
                // threads wrote in pass 1. This barrier is load-bearing.
                #pragma omp barrier
                w2 = prof ? omp_get_wtime() : 0.0;

                #pragma omp for schedule(dynamic, 64) nowait
                for (long t = 0; t < nRuns; t++) {
                    const long p0 = runs[t].p;
                    const int len = runs[t].len;
                    for (int m = 0; m < len; m += RBLK) {
                        const int n = (len - m < RBLK) ? (len - m) : RBLK;
                        double pre[RBLK], post[RBLK];
                        for (int c = 0; c < n; c++)
                            pre[c] = post[c] = u[p0 + m + c];
                        reactBlock(post, n);
                        // Ascending c inside ascending m is the order the
                        // one-cell-at-a-time version used, so this thread's
                        // partial sum is unchanged.
                        for (int c = 0; c < n; c++) {
                            u[p0 + m + c] = post[c];
                            e += post[c] * post[c] - pre[c] * pre[c];
                        }
                    }
                }
                w3 = prof ? omp_get_wtime() : 0.0;
            }

            // Nothing below needs another thread's writes, so the region's own
            // closing barrier is the last one required. The barrier here exists
            // only to separate the wait from the work in the profile, so it is
            // taken only when the profile is being collected.
            if (prof) {
                #pragma omp barrier
                const double w4 = omp_get_wtime();
                tp[0] += (w1 - w0) * 1000.0;   // pass 1 work
                tp[1] += (w2 - w1) * 1000.0;   // wait at the barrier after 1
                tp[2] += (w3 - w2) * 1000.0;   // pass 2 work
                tp[3] += (w4 - w3) * 1000.0;   // wait at the barrier after 2
            }
        }

        steps++;
        energy = e;
        const double s1 = nowMs();
        tStencilAcc += s1 - s0;
        if (energy <= theta * energy0) break;
    }

    const double tSteps = nowMs();

    FILE* out = fopen(argv[5], "w");
    if (!out) {
        perror(argv[5]);
        return 1;
    }
    // Answer (1): the steps taken and the energy left in the block.
    fprintf(out, "%ld %d\n%.17g\n", N, steps, energy);

    // Answer (2): print the 32 x 32 x 32 lattice of sample temperatures.
    for (int si = 0; si < 32; si++)
        for (int sj = 0; sj < 32; sj++)
            for (int sk = 0; sk < 32; sk++) {
                const long i = 1 + si * (N - 1) / 31, j = 1 + sj * (N - 1) / 31, k = 1 + sk * (N - 1) / 31;
                fprintf(out, "%.17g\n", u[i * SI + j * SJ + k]);
            }
    fclose(out);

    if (prof) {
        const double tEnd = nowMs();
        fprintf(stderr,
                "threads %2d | N %ld M %ld | T %d steps %d | arrays %.2f GB\n"
                "  alloc %8.1f | gen %9.1f | energy0 %7.1f | "
                "stencil+energy %9.1f | out %7.1f | total %9.1f  (ms)\n",
                omp_get_max_threads(), N, M, T, steps,
                (double)cells * BYTES_PER_CELL / (1024.0 * 1024.0 * 1024.0),
                tAlloc - tStart, tGen - tAlloc, tEnergy0 - tGen,
                tStencilAcc, tEnd - tSteps, tEnd - tStart);

        // max against mean is the point: a barrier whose max is well above its
        // mean means imbalance rather than slow work.
        static const char* slotName[TPROF_SLOTS] = {
            "pass1 stencil", "barrier after 1", "pass2 react", "barrier after 2"};
        fprintf(stderr, "  per-thread, summed over steps (ms)\n");
        fprintf(stderr, "    %-16s %9s %9s %9s %9s\n",
                "", "min", "mean", "max", "max/mean");
        for (int s = 0; s < TPROF_SLOTS; s++) {
            double sum = 0.0, mx = 0.0, mn = 1e300;
            for (int t = 0; t < nthreads; t++) {
                const double v = tprof[(size_t)t * TPROF_STRIDE + s];
                sum += v;
                if (v > mx) mx = v;
                if (v < mn) mn = v;
            }
            const double mean = sum / nthreads;
            fprintf(stderr, "    %-16s %9.1f %9.1f %9.1f %9.2f\n",
                    slotName[s], mn, mean, mx, mean > 0.0 ? mx / mean : 0.0);
        }
    }

    free(u);
    free(a);
    free(planeBufs);
    free(firstPlanes);
    free(rowBufs);
    return 0;
}
