# HW 2-1: shortest paths on a distributed graph with MPI

The starter computes exact shortest paths from each of S sources on a generated graph, with
Dijkstra on rank 0. Make it fast on 24 single-core MPI ranks on one node. The starter exceeds
the memory limit on the larger cases, so no rank can hold the whole graph.
HW 2-1 is 50% of HW 2, scored 70 correctness / 30 performance.

## Files

| file | status | what it is |
|---|---|---|
| `main.cpp` | fixed | `main()`: reads the input, calls your `run()`, writes the output |
| `graph_gen.hpp` | fixed | the graph generator's interface. Read this first. |
| `graph_gen.cpp` | fixed | the graph generator |
| `hw2-1.cpp` | yours | `run()`: your MPI program; the file you submit, plus any header of your own it includes |
| `Makefile` | provided | the judge's build, for your own machine |

The judge compiles your files against its own copies of the fixed files, with
`mpicxx -std=c++17 -O3 -march=native`. Editing a fixed file or the flags changes nothing on
the judge.

## Build and run

```sh
module load gcc/15.2.0 openmpi/5.0.10
make
srun -N1 -n24 -c1 --hint=nomultithread ./hw2-1 cases/hw2-1/01.txt out.bin
hw2-1-check cases/hw2-1/01.golden.bin out.bin   # zero on a match
```

The judge runs exactly this shape: 24 ranks on one node, one physical core each, 48 GiB on
the node and 1200 MiB of private data per rank (`RLIMIT_DATA`). An allocation past the cap
throws `std::bad_alloc`. MPI shared-memory windows are exempt and count against the node's
48 GiB instead. Run `ulimit -d 1228800` before `srun` to reproduce the cap.

`Compute time` runs from a barrier after every rank has read the input until `run()` has
returned on every rank. Graph generation counts.

## The graph

`graph_gen.hpp` provides:

- `generate_edge(p, i, u, v, w)`: edge `i`, `0 <= i < M`, as a pure function of `p` and `i`;
- `generate_edges(p, i0, i1, u, v, w)`: edges `i0..i1-1` into three arrays;
- `sample_vertices(p)`: the vertices whose distances the output lists;
- `distance_digest(v, d, k)`: one vertex's term of digest `k`.

Any rank can generate any subset of the edges, in any order, without communication. The graph
is an undirected Kronecker graph with `N = 2^scale` vertices, `M = edgefactor * N` edges and
weights in `[1, 255]`: a few vertices carry a large share of the edges, and many are isolated.
With `scramble = 1` a fixed bijection permutes the vertex ids before weights are assigned, so
the two settings are different weighted graphs. Self-loops and repeated edges occur; you may
skip them. Generate the graph in memory; do not write or read edges from a file.

## Input and output

The input is one line: `scale edgefactor seed scramble nsample S src_1 ... src_S`.
For each source, fill on rank 0, in input order:

| field | value |
|---|---|
| `n_reached` | reachable vertices, the source included |
| `checksum` | sum of `d[v] * (v + 1)` over reachable `v`, mod 2^64 |
| `digest1`, `digest2` | sum of `distance_digest(v, d[v], k)` over reachable `v`, mod 2^64, for `k = 0, 1`; the source counts at distance 0 |
| `sample_dist` | distances in `sample_vertices(p)` order, `-1` if unreachable |

`main.cpp` writes the binary output; other ranks' results are ignored. Finish every MPI
operation before returning, and do not redeclare `SourceResult`.

## Input constraints

| field | range |
|---|---|
| scale | 1..26 |
| edgefactor | 1..256, with M <= 2^30 |
| seed | 0..2^64-1 |
| scramble | 0 or 1 |
| nsample | 0..N if scale <= 16, otherwise 0..1000 |
| S | 1..32 |
| each source | 0..N-1 |

Every finite distance fits `int32_t`; use wider arithmetic for tentative distances and
checksums. Sources may repeat or be isolated: return one result per occurrence. An isolated
source reaches only itself. Samples are distinct and sorted. With zero samples, still fill
the other fields. A rank may own no vertices or edges.

## Develop at small scale

Keep the starter's executable; it produces the reference output for any case you write:

```sh
make && cp hw2-1 hw2-1-seq
echo "16 16 1 1 100 2 0 12345" > small.txt
srun -n 1 ./hw2-1-seq small.txt small.golden.bin
srun -n 4 ./hw2-1 small.txt out.bin
hw2-1-check small.golden.bin out.bin
```

## Rules

- No threads of any kind (OpenMP, `std::thread`, pthreads, `std::async`, parallel STL), no
  GPUs, no third-party libraries. C++17, its standard library, MPI, MPI shared-memory windows
  and compiler SIMD intrinsics are allowed.
- Read only the input file and write only the output file.
- No answers embedded in the code or looked up by case identity.

Breaking a rule scores 0 on the problem. The judge reads your source, the binary's imports
and your ranks' thread counts.
