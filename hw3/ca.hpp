// HW 3-1: a Larger than Life cellular automaton with Generations decay.
//
// FIXED FILE. The judge compiles its own copy; changes to yours have no effect.
//
// The grid is W x H cells, each holding a state in [0, C):
//
//   0          dead
//   1          alive          -- the only state that counts as a live neighbour
//   2 .. C-1   decaying       -- a live cell that failed to survive, on its way back to 0
//
// One generation updates every cell at once, from the grid as it was:
//
//   n = the number of cells in state 1 in the cell's neighbourhood
//
//   state 0:      becomes 1 if birth_lo <= n <= birth_hi, else stays 0
//   state 1:      stays 1 if survive_lo <= n <= survive_hi, else becomes 2 (0 when C == 2)
//   state k >= 2: becomes k + 1, and 0 after C - 1, whatever n is
//
// The neighbourhood of (x, y) is every (x + dx, y + dy) with
//
//   Moore:        max(|dx|, |dy|) <= R
//   von Neumann:  |dx| + |dy| <= R
//
// and the cell itself, (dx, dy) = (0, 0), only when count_centre is set. A
// neighbour outside the grid wraps around on a torus and is dead (not counted)
// on a dead boundary.
//
// A test case writes the rule as R<R>,C<C>,M<0|1>,S<lo>..<hi>,B<lo>..<hi>,N<M|N>.

#pragma once

#include <cstdint>

constexpr int MAX_RADIUS = 32;
constexpr int MAX_STATES = 16;

enum class Shape : uint8_t { Moore, VonNeumann };
enum class Boundary : uint8_t { Torus, Dead };

struct Rule {
    int radius;          // R, 1 .. MAX_RADIUS
    int states;          // C, 2 .. MAX_STATES
    bool count_centre;   // M1: the cell counts itself when it is alive
    // With maxn the neighbourhood's size without the cell itself ((2R+1)^2 - 1 for Moore,
    // 2R(R+1) for von Neumann) and m = 1 under M1, else 0:
    int survive_lo, survive_hi;   // m <= survive_lo <= survive_hi <= maxn + m
    int birth_lo, birth_hi;       // 1 <= birth_lo <= birth_hi <= maxn + m: an empty region stays empty
    Shape shape;
};

struct Problem {
    int width, height;   // W, H; 2R + 1 <= min(W, H)
    Rule rule;
    Boundary boundary;
    int steps;           // T, the number of generations
};

// Advances `grid` by p.steps generations, in place.
//
// grid: p.width * p.height cells, row-major (cell (x, y) at grid[y * width + x]),
// one byte per cell holding its state. It is pinned host memory, allocated by
// main() with cudaMallocHost, and on return it must hold the final generation.
//
// main() times this call, from entry until it has returned and the device is
// synchronized. Every generation has to be computed on the GPU: host code may
// allocate, copy, launch kernels and decide what to launch, but compute no part
// of any generation.
void run(uint8_t *grid, const Problem &p);
