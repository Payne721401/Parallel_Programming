// HW 2-1: YOUR FILE. All of your MPI goes here, in run(), the function main() calls.
//
// This starting version is correct and sequential: process 0 generates the whole graph, builds
// it in memory and runs Dijkstra from every source; the other processes do nothing. The starting
// version works at small scale on your own machine and fails on the judge: at the scales of the
// test cases the graph does not fit one process's 1.2 GB (scale 24 is about 2.6 GB of adjacency),
// Construction temporaries already exceed the cap at scale 22, edgefactor 16,
// and one core is minutes over the large-case time limits. Distributing the graph and the search is the
// assignment.
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "graph_gen.hpp"


namespace {

// Compressed sparse rows of the whole (undirected) graph: every edge in both directions.
struct CSR {
    std::vector<uint64_t> row;      // [N + 1]
    std::vector<uint32_t> col;      // [2M]
    std::vector<uint8_t> w;         // [2M]
};

CSR build_whole_graph(const GraphParams &p)
{
    const uint64_t N = num_vertices(p), M = num_edges(p);
    std::vector<uint32_t> u(M), v(M);
    std::vector<uint8_t> w(M);
    generate_edges(p, 0, M, u.data(), v.data(), w.data());
    CSR g;
    g.row.assign(N + 1, 0);
    for (uint64_t i = 0; i < M; i++) { g.row[u[i] + 1]++; g.row[v[i] + 1]++; }
    for (uint64_t x = 0; x < N; x++) g.row[x + 1] += g.row[x];
    g.col.resize(2 * M);
    g.w.resize(2 * M);
    std::vector<uint64_t> fill(g.row.begin(), g.row.end() - 1);
    for (uint64_t i = 0; i < M; i++) {
        g.col[fill[u[i]]] = v[i]; g.w[fill[u[i]]++] = w[i];
        g.col[fill[v[i]]] = u[i]; g.w[fill[v[i]]++] = w[i];
    }
    return g;
}

const uint32_t INF = 0xffffffffu;

void dijkstra(const CSR &g, uint32_t src, std::vector<uint32_t> &dist)
{
    dist.assign(g.row.size() - 1, INF);
    using QE = std::pair<uint32_t, uint32_t>;   // (distance, vertex)
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> q;
    dist[src] = 0;
    q.push({0, src});
    while (!q.empty()) {
        auto [d, x] = q.top();
        q.pop();
        if (d != dist[x]) continue;
        for (uint64_t e = g.row[x]; e < g.row[x + 1]; e++) {
            const uint32_t y = g.col[e], nd = d + g.w[e];
            if (nd < dist[y]) {
                dist[y] = nd;
                q.push({nd, y});
            }
        }
    }
}

}  // namespace

void run(const GraphParams &p, const std::vector<uint32_t> &sources, std::vector<SourceResult> &results)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;   // <- every other process idles. Start by changing this line.

    const CSR g = build_whole_graph(p);
    const std::vector<uint32_t> samples = sample_vertices(p);
    std::vector<uint32_t> dist;
    for (size_t s = 0; s < sources.size(); s++) {
        dijkstra(g, sources[s], dist);
        SourceResult &r = results[s];
        r.n_reached = 0;
        r.checksum = 0;
        r.digest1 = r.digest2 = 0;
        for (uint64_t v = 0; v < dist.size(); v++)
            if (dist[v] != INF) {
                r.n_reached++;
                r.checksum += (uint64_t)dist[v] * (v + 1);
                r.digest1 += distance_digest(v, dist[v], 0);
                r.digest2 += distance_digest(v, dist[v], 1);
            }
        for (int k = 0; k < p.nsample; k++) r.sample_dist[k] = dist[samples[k]] == INF ? -1 : (int32_t)dist[samples[k]];
    }
}
