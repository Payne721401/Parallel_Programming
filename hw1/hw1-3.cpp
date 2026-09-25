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
#include <vector>

// ========== START: DO NOT CHANGE BELOW ==========
static const double R = 0.5;    // the reaction's strength
static const int SUBSTEPS = 4;  // sub-steps of the reaction per time step
static const int NEWTON = 3;    // Newton iterations per sub-step

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
// =========== END: DO NOT CHANGE ABOVE ===========

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

    // (N+2)^3 with a halo of zeros around the block, so no cell is a special case.
    const long M = N + 2;
    const long SI = M * M, SJ = M;  // strides of i and j; k is contiguous
    std::vector<double> u(M * M * M, 0.0), unew(M * M * M, 0.0), a(M * M * M, 0.0);
    std::vector<uint8_t> mat(M * M * M, 0);  // 1 where the cell is reactive

    Inclusion inc[4];
    const int B = inclusions(seed, N, inc);
    for (long i = 1; i <= N; i++)
        for (long j = 1; j <= N; j++)
            for (long k = 1; k <= N; k++) {
                const long p = i * SI + j * SJ + k;
                const uint64_t index = ((i - 1) * N + (j - 1)) * N + (k - 1);
                u[p] = field(seed, index, 0);
                a[p] = field(seed, index, 1);
                mat[p] = reactive(inc, B, i, j, k);
            }

    // The energy of the block: the sum of u^2 over its cells.
    auto energy_of = [&](const std::vector<double>& v) {
        double energy = 0.0;
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
    while (steps < T) {
        for (long i = 1; i <= N; i++)
            for (long j = 1; j <= N; j++)
                for (long k = 1; k <= N; k++) {
                    const long p = i * SI + j * SJ + k;
                    const double up = u[p], ap = a[p];
                    double flux = 0.0;
                    flux += (ap + a[p - SI]) * (u[p - SI] - up);
                    flux += (ap + a[p + SI]) * (u[p + SI] - up);
                    flux += (ap + a[p - SJ]) * (u[p - SJ] - up);
                    flux += (ap + a[p + SJ]) * (u[p + SJ] - up);
                    flux += (ap + a[p - 1]) * (u[p - 1] - up);
                    flux += (ap + a[p + 1]) * (u[p + 1] - up);
                    const double r = up + flux * (1.0 / 12.0);
                    unew[p] = mat[p] ? react(r) : r;
                }
        u.swap(unew);
        steps++;
        energy = energy_of(u);
        if (energy <= theta * energy0) break;
    }

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
    return 0;
}
