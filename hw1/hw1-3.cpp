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
#include <utility>
#include <vector>
#include <chrono>
#include <omp.h>
#include <sched.h>

// Set -DSTENCIL_SIMD=0 to drop the omp simd directive on the diffusion loop and
// let GCC's cost model decide instead. The stencil is memory-bound at roughly
// 1 flop per byte, so forcing vectorisation may well be a pessimisation.
#ifndef STENCIL_SIMD
#define STENCIL_SIMD 1
#endif

// How many cells reactBatch() advances at once. Sweep with -DREACT_BATCH=8.
#ifndef REACT_BATCH
#define REACT_BATCH 4
#endif

// Per-thread profile counters, one 64-byte line each so the threads never share
// a cache line while accumulating. Slots: pass 1 work, wait at the barrier after
// pass 1, pass 2 work, wait at the barrier after pass 2.
#define TPROF_STRIDE 8
#define TPROF_SLOTS 4

// ---------- staged timing ----------
// Gated on PP_TIMING so the judge run stays clean:
//   PP_TIMING=1 srun -n 1 -c 8 ./hw1-3 768 6 1087 0 out.txt
static inline double nowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Section 1.4 of the spec: "Use sched_getaffinity and CPU_COUNT to determine
// how many cores are available to your program, and create that many threads."
// Public cases run with 2, 4 or 8 cores, and the cluster overwrites
// OMP_NUM_THREADS to 1, so the affinity mask is the only reliable source.
static int usableCpus() {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
    return omp_get_max_threads();
}

// These sit outside the protected range as of the 2026-09-29 handout, but the
// spec defines the reference answer by this exact iteration count rather than
// by the converged root, so changing the values still changes the answer.
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

// reactive() and react() below are ours to change now. react() in particular:
// it is 12 exp() calls deep in a single dependency chain, one cell at a time.

// A maximal run of consecutive reactive cells along k, as a flat index and a
// length. Inclusions are spheres, so for a given (i, j) each one contributes at
// most one contiguous range of k and the union of at most four of them is at
// most four disjoint runs. Holding the reactive set this way costs about 3 MB
// at N=960 against 890 MB for a full per-cell mat array, and it drops a read
// stream from every time step.
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
static double react(double r) {
    double x = r;
    for (int s = 0; s < SUBSTEPS; s++) {
        const double prev = x;
        for (int n = 0; n < NEWTON; n++) {
            const double e = exp(x);
            x -= (x + R * (e - 1.0) - prev) / (1.0 + R * e);
        }
    }
    return x;
}

// react() advanced for REACT_BATCH cells together, in place.
//
// The scalar version is SUBSTEPS * NEWTON = 12 exp() calls where each one waits
// on the previous call's result, so almost all of its cost is that chain:
// measured at about 80 cycles per exp against exp's own ~15-cycle throughput.
// Interleaving several cells puts that many independent chains in flight, and
// the out-of-order window (hundreds of instructions, against roughly 30 for one
// exp) is wide enough to overlap them.
//
// Every cell still walks the identical sequence -- prev = x, then NEWTON rounds
// of exp and update, SUBSTEPS times, with the same expression and the same
// operands -- so each result is bit-identical to react(). Only the interleaving
// changes, and interleaving independent chains is not a reordering of any one
// chain.
//
// Note what is deliberately absent: no omp simd on the exp() loop. That would
// let GCC call glibc's vector exp, which is allowed 4 ULP against the scalar
// routine's sub-ULP. FMA's roughly 1 ULP was already enough to turn p01 and p03
// into WA, so 4 ULP is out of the question.
static void reactBatch(double* x) {
    for (int s = 0; s < SUBSTEPS; s++) {
        double prev[REACT_BATCH];
        for (int v = 0; v < REACT_BATCH; v++) prev[v] = x[v];
        for (int n = 0; n < NEWTON; n++) {
            double ex[REACT_BATCH];
            for (int v = 0; v < REACT_BATCH; v++) ex[v] = exp(x[v]);
            for (int v = 0; v < REACT_BATCH; v++)
                x[v] -= (x[v] + R * (ex[v] - 1.0) - prev[v]) / (1.0 + R * ex[v]);
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
    // calloc, not std::vector: a vector value-initialises, which memsets 11 GB
    // of zeros at N=768 on one thread before a single cell is computed. calloc
    // on an allocation this size gets anonymous pages from the kernel that are
    // already zero, so it skips the memset entirely and the pages materialise
    // on first touch -- inside the parallel generation loop below, which also
    // places each page on the NUMA node of the thread that will keep using it.
    // The halo stays zero either way, so the boundary semantics are unchanged.
    const size_t cells = (size_t)M * M * M;
    // u + unew + a, all double. The reactive set used to be a fourth array of
    // one byte per cell; it is a list of runs now. Both the PP_TIMING line and
    // the out-of-memory message report through this, so keep it in step when an
    // array is added or removed.
    const double BYTES_PER_CELL = 24.0;
    double* u = (double*)calloc(cells, sizeof(double));
    double* unew = (double*)calloc(cells, sizeof(double));
    double* a = (double*)calloc(cells, sizeof(double));
    if (!u || !unew || !a) {
        fprintf(stderr, "out of memory for N=%ld (%.2f GB)\n",
                N, cells * BYTES_PER_CELL / (1024.0 * 1024.0 * 1024.0));
        return 1;
    }

    const double tAlloc = nowMs();

    Inclusion inc[4];
    const int B = inclusions(seed, N, inc);

    // field() is a pure function of the index, so every cell is independent --
    // no sequential PRNG state to carry. Running it in parallel also first-
    // touches the pages from the thread that will keep using them.
    //
    // The same sweep records the reactive set. It already evaluates reactive()
    // for every cell, so tracking where the predicate turns on and off along k
    // is free, and the resulting runs replace the per-cell mat array entirely.
    // Each thread owns whole i planes, so the per-plane buffers need no locking,
    // and concatenating them in i order afterwards keeps the list deterministic.
    std::vector<std::vector<Run> > runsPerPlane(N + 1);

    #pragma omp parallel for schedule(static)
    for (long i = 1; i <= N; i++) {
        std::vector<Run>& out = runsPerPlane[i];
        for (long j = 1; j <= N; j++) {
            long runStart = -1;
            for (long k = 1; k <= N; k++) {
                const long p = i * SI + j * SJ + k;
                const uint64_t index = ((i - 1) * N + (j - 1)) * N + (k - 1);
                u[p] = field(seed, index, 0);
                a[p] = field(seed, index, 1);
                if (reactive(inc, B, i, j, k)) {
                    if (runStart < 0) runStart = p;
                } else if (runStart >= 0) {
                    out.push_back(Run{runStart, (int)(p - runStart)});
                    runStart = -1;
                }
            }
            if (runStart >= 0) {   // the run reached the far face
                const long pEnd = i * SI + j * SJ + (N + 1);
                out.push_back(Run{runStart, (int)(pEnd - runStart)});
            }
        }
    }

    std::vector<Run> runs;
    for (long i = 1; i <= N; i++)
        runs.insert(runs.end(), runsPerPlane[i].begin(), runsPerPlane[i].end());
    std::vector<std::vector<Run> >().swap(runsPerPlane);   // give the memory back
    const long nRuns = (long)runs.size();

    const double tGen = nowMs();

    // The energy of the block: the sum of u^2 over its cells.
    // Only used for energy0 now; the per-step energy is fused into the stencil
    // loop below so the array is swept once per step instead of twice.
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

    // Totals alone hide imbalance: eight threads averaging 100 ms tell you
    // nothing about whether one of them took 400 ms while the rest waited. Time
    // each pass and each barrier per thread instead, and report max against
    // mean -- the barrier columns are where imbalance actually shows up.
    std::vector<double> tprof((size_t)nthreads * TPROF_STRIDE, 0.0);

    while (steps < T) {
        const double s0 = nowMs();

        // __restrict: these are three separate calloc blocks, but as plain
        // double* the compiler has to assume they might overlap, and that alone
        // stops the k loop from vectorising.
        const double* __restrict U = u;
        const double* __restrict A = a;
        double* __restrict UN = unew;

        double e = 0.0;

        // One parallel region for both passes, with nowait on each omp for and
        // an explicit barrier after it, so the wait can be timed separately
        // from the work.
        #pragma omp parallel reduction(+ : e)
        {
            double* tp = &tprof[(size_t)omp_get_thread_num() * TPROF_STRIDE];
            const double w0 = omp_get_wtime();

            // Pass 1: plain diffusion for every cell. No branch and no call in
            // the k loop now that the reaction has moved out, and static,
            // because with react() gone every cell costs the same. (It needed
            // dynamic before: reactive cells run ~30x a plain one and sit
            // inside spheres spanning only ~0.3 N planes, so a fixed split left
            // most threads idle.)
            //
            // omp simd is what would make the reduction legal to vectorise:
            // e += r * r needs its additions regrouped, and GCC will not do
            // that unprompted without -ffast-math, which we cannot afford
            // globally. Whether it is worth doing is another question -- see
            // STENCIL_SIMD. The temperatures are unaffected either way, since
            // every UN[p] is the same expression evaluated in the same order.
            #pragma omp for schedule(static) nowait
            for (long i = 1; i <= N; i++)
                for (long j = 1; j <= N; j++) {
                    const long base = i * SI + j * SJ;
#if STENCIL_SIMD
                    #pragma omp simd reduction(+ : e)
#endif
                    for (long k = 1; k <= N; k++) {
                        const long p = base + k;
                        const double up = U[p], ap = A[p];
                        double flux = 0.0;
                        flux += (ap + A[p - SI]) * (U[p - SI] - up);
                        flux += (ap + A[p + SI]) * (U[p + SI] - up);
                        flux += (ap + A[p - SJ]) * (U[p - SJ] - up);
                        flux += (ap + A[p + SJ]) * (U[p + SJ] - up);
                        flux += (ap + A[p - 1]) * (U[p - 1] - up);
                        flux += (ap + A[p + 1]) * (U[p + 1] - up);
                        const double r = up + flux * (1.0 / 12.0);
                        UN[p] = r;
                        e += r * r;
                    }
                }

            const double w1 = omp_get_wtime();
            #pragma omp barrier
            const double w2 = omp_get_wtime();

            // Pass 2: replace the plain value in each reactive cell with the
            // reacted one. Pass 1 already added pre^2 to the energy, so adding
            // the difference costs O(reactive cells) instead of another full
            // sweep. dynamic: a run is a chord through a sphere, so lengths
            // vary from 1 to twice the radius.
            #pragma omp for schedule(dynamic, 64) nowait
            for (long t = 0; t < nRuns; t++) {
                const long p0 = runs[t].p;
                const int len = runs[t].len;
                int m = 0;
                for (; m + REACT_BATCH <= len; m += REACT_BATCH) {
                    double x[REACT_BATCH], pre[REACT_BATCH];
                    for (int v = 0; v < REACT_BATCH; v++)
                        pre[v] = x[v] = UN[p0 + m + v];
                    reactBatch(x);
                    for (int v = 0; v < REACT_BATCH; v++) {
                        UN[p0 + m + v] = x[v];
                        e += x[v] * x[v] - pre[v] * pre[v];
                    }
                }
                for (; m < len; m++) {   // tail of a run shorter than a batch
                    const long p = p0 + m;
                    const double pre = UN[p];
                    const double post = react(pre);
                    UN[p] = post;
                    e += post * post - pre * pre;
                }
            }

            const double w3 = omp_get_wtime();
            #pragma omp barrier
            const double w4 = omp_get_wtime();

            tp[0] += (w1 - w0) * 1000.0;   // pass 1 work
            tp[1] += (w2 - w1) * 1000.0;   // waiting at the barrier after pass 1
            tp[2] += (w3 - w2) * 1000.0;   // pass 2 work
            tp[3] += (w4 - w3) * 1000.0;   // waiting at the barrier after pass 2
        }

        std::swap(u, unew);
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

    if (getenv("PP_TIMING")) {
        const double tEnd = nowMs();
        fprintf(stderr,
                "threads %2d | N %ld M %ld | T %d steps %d | arrays %.2f GB\n"
                "  alloc %8.1f | gen %9.1f | energy0 %7.1f | "
                "stencil+energy %9.1f | out %7.1f | total %9.1f  (ms)\n",
                omp_get_max_threads(), N, M, T, steps,
                (double)cells * BYTES_PER_CELL / (1024.0 * 1024.0 * 1024.0),
                tAlloc - tStart, tGen - tAlloc, tEnergy0 - tGen,
                tStencilAcc, tEnd - tSteps, tEnd - tStart);

        // max against mean is the point: a barrier column whose max is well
        // above its mean means the threads finished the preceding pass at very
        // different times, which is imbalance rather than slow work.
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
    free(unew);
    free(a);
    return 0;
}
