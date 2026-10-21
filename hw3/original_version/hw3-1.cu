// HW 3-1: your implementation goes in this file.
//
// As handed out, run() is the sequential reference: correct, on one CPU core,
// and never touching the GPU. It is far too slow for anything but the smallest
// cases, and it breaks the rule that every generation is computed on the GPU
// (see ca.hpp). Replace it.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "ca.hpp"

// Whether (dx, dy) is in the neighbourhood, centre included only under M1.
static bool in_neighbourhood(const Rule &r, int dx, int dy) {
    if (dx == 0 && dy == 0) return r.count_centre;
    if (r.shape == Shape::Moore) return true;   // the loops below already bound max(|dx|, |dy|)
    return std::abs(dx) + std::abs(dy) <= r.radius;
}

// The state of neighbour (x, y), or -1 when it lies outside a dead boundary.
static int neighbour(const uint8_t *grid, const Problem &p, int x, int y) {
    if (p.boundary == Boundary::Torus) {
        x = (x + p.width) % p.width;   // 2R + 1 <= W: one wrap is enough
        y = (y + p.height) % p.height;
    } else if (x < 0 || x >= p.width || y < 0 || y >= p.height) {
        return -1;
    }
    return grid[size_t(y) * p.width + x];
}

static uint8_t next_state(const Rule &r, uint8_t s, int n) {
    if (s == 0) return (r.birth_lo <= n && n <= r.birth_hi) ? 1 : 0;
    if (s == 1) return (r.survive_lo <= n && n <= r.survive_hi) ? 1 : 2 % r.states;
    return (s + 1) % r.states;
}

void run(uint8_t *grid, const Problem &p) {
    const Rule &r = p.rule;
    std::vector<uint8_t> next(size_t(p.width) * p.height);

    for (int t = 0; t < p.steps; t++) {
        for (int y = 0; y < p.height; y++) {
            for (int x = 0; x < p.width; x++) {
                int n = 0;
                for (int dy = -r.radius; dy <= r.radius; dy++)
                    for (int dx = -r.radius; dx <= r.radius; dx++)
                        if (in_neighbourhood(r, dx, dy) && neighbour(grid, p, x + dx, y + dy) == 1)
                            n++;
                next[size_t(y) * p.width + x] = next_state(r, grid[size_t(y) * p.width + x], n);
            }
        }
        std::copy(next.begin(), next.end(), grid);
    }
}
