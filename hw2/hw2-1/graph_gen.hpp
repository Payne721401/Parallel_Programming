// HW 2-1: the graph. BASE FILE: the judge builds with its own copy, so editing yours changes
// nothing there. Read this file for what the functions give you.
//
// The input names the graph; your program generates it, edge by edge, from the few numbers on
// that line. Edge i is a pure function of (p, i), so any process can generate any subset of the
// edges on its own. A process generates the edges it wants, and nobody has to hold the whole
// graph. Your program's first decision is which edges each process generates, and where the
// edges that touch vertices it does not own go.
//
// Kronecker (R-MAT) graph, as in Graph500: N = 2^scale vertices, M = edgefactor * N undirected
// edges. Each edge starts at the (0, 0) corner of the adjacency matrix and descends `scale`
// levels, choosing a quadrant with probabilities A = 0.57, B = 0.19, C = 0.19, D = 0.05, so a
// few vertices end up with enormous degree and most with almost none. With scramble = 0 the
// vertex ids are the raw Kronecker ids (the heavy vertices sit at low ids); with scramble = 1
// a fixed bijection permutes them. Self-loops and repeated edges occur and are part of
// the graph; they change no shortest path. Every edge has an integer weight in [1, 255].
#ifndef GRAPH_GEN_HPP
#define GRAPH_GEN_HPP

#include <cstdint>
#include <vector>

// Valid input contract (checked by main before run):
// 1 <= scale <= 26; 1 <= edgefactor <= 256; edgefactor * N <= 2^30;
// seed is any uint64_t; scramble is 0 or 1; 0 <= nsample <= N for scale <= 16; otherwise <= 1000.
// There are 1..32 sources in [0, N); sources may repeat or be isolated.
// All generator calls must satisfy these preconditions and their edge-index bounds.
struct GraphParams {
    int scale;            // N = 1 << scale vertices
    int edgefactor;       // M = edgefactor * N edges
    uint64_t seed;
    int scramble;         // 0: raw Kronecker ids, 1: permuted ids
    int nsample;          // vertices whose distances the output lists
};

inline uint64_t num_vertices(const GraphParams &p) { return uint64_t(1) << p.scale; }
inline uint64_t num_edges(const GraphParams &p) { return uint64_t(p.edgefactor) << p.scale; }

// Edge i of the graph, 0 <= i < num_edges(p): its endpoints (after scrambling, if on) and its
// weight. Deterministic; the same (p, i) gives the same edge on every process, every run.
void generate_edge(const GraphParams &p, uint64_t i, uint32_t &u, uint32_t &v, uint8_t &w);

// Edges [i0, i1) into three arrays of i1 - i0 entries. The same as calling generate_edge in a
// loop.
void generate_edges(const GraphParams &p, uint64_t i0, uint64_t i1, uint32_t *u, uint32_t *v, uint8_t *w);

// The nsample distinct vertices whose distances go into the output, ascending. A fixed function
// of the parameters, so the judge and your program agree on it. Empty when nsample == 0.
std::vector<uint32_t> sample_vertices(const GraphParams &p);

// All arithmetic below is unsigned modulo 2^64. Hash only finite distances.
// Domain-separated nonlinear fingerprints supplement the linear checksum; they are
// deterministic error detectors, not cryptographic proofs of a complete distance vector.
inline uint64_t distance_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
inline uint64_t distance_digest(uint32_t vertex, uint32_t distance, unsigned which) {
    const uint64_t key = (uint64_t(vertex) << 32) | uint64_t(distance);
    return distance_mix(key ^ (which == 0 ? 0x243f6a8885a308d3ULL : 0x13198a2e03707344ULL));
}
struct SourceResult {
    int64_t n_reached = 0;
    uint64_t checksum = 0;
    uint64_t digest1 = 0, digest2 = 0;
    std::vector<int32_t> sample_dist;
};

#endif
