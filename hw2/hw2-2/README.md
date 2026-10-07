# HW 2-2: sparse matrix times sparse matrix with MPI

Compute the exact product C = AB of two sparse integer matrices, M-by-K A and K-by-N B. Make
it fast on 24 single-core MPI ranks across two nodes. The starter gives every rank a private
copy of B and fails the performance cases by design. HW 2-2 is 50% of HW 2, scored
40 correctness / 60 performance. `SPEC.md` holds the full limits and the binary CSR layout.

## Files

| file | status | what it is |
|---|---|---|
| `main.cpp` | fixed | `main()`: reads the inputs, times your `multiply()`, validates and writes the result |
| `spgemm.hpp` | fixed | the `Matrix` layout and the ownership helpers. Read this first. |
| `local_kernel.hpp` | fixed | an optional accumulator that builds one sorted output row |
| `hw2-2.cpp` | yours | `multiply()`: your MPI program; the file you submit |
| `Makefile` | provided | the judge's build, for your own machine |

The judge compiles your file against its own copies of the fixed files, with
`mpicxx -std=c++17 -O3 -march=native`. Editing a fixed file or the flags changes nothing on
the judge.

## Build and run

```sh
module load gcc/15.2.0 openmpi/5.0.10
make
srun -N2 -n24 --ntasks-per-node=12 --hint=nomultithread \
    ./hw2-2 cases/hw2-2/c01/A.csr cases/hw2-2/c01/B.csr out.csr
hw2-2-check cases/hw2-2/c01/C.csr out.csr   # zero on a match, else the first difference
```

The judge runs exactly this shape: 12 ranks on each of two nodes, one physical core each,
48 GiB per node and 1200 MiB of private data per rank (`RLIMIT_DATA`). B alone is 2.3 to
2.9 GB on a performance case, so no rank can hold its own copy. MPI shared-memory windows are
exempt from the cap and count against the node's 48 GiB instead.

`Compute time` covers the whole `multiply()` call and its completion barrier, on the slowest
rank. Loading, validation and writing are outside it but count toward the 60 s wall limit.
The compute limit is 10 s on a correctness case and 5 s on a performance case. The result must
not depend on the rank count, so `srun -n4 ./hw2-2 ...` is a fair test on every case.

## Interface

```cpp
void multiply(const Matrix& a, const Matrix& b, Matrix& c);
```

- With P ranks, rank r receives A rows `[floor(M*r/P), floor(M*(r+1)/P))` and B rows with the
  analogous K bounds. A rank may hold no rows.
- `rows` and `cols` are global dimensions, `first` the first owned global row. `ptr` indexes
  the local `data` from zero; a rank with no rows has `ptr = {0}`.
- Return in `c` the C rows of your original A rows: dimensions M, N and `c.first = a.first`,
  each row's columns sorted and unique, values exact and positive.
- Do not write to the input shards. You may redistribute data and use MPI shared-memory
  windows; discover node-local groups through MPI. Finish every MPI operation before returning.

## Input constraints

| quantity | allowed values |
|---|---|
| M, N | 1..8192 |
| K | 1..4,194,304 |
| stored values | integers in [1, 8] |
| columns within a row | strictly increasing |
| output values | fit `uint32_t` |
| work and size counts | may exceed 2^32: count them in 64 bits |
| initial A + B per rank, at 24 ranks | at most 512 MiB of CSR storage |
| final C per rank, at 24 ranks | at most 32 MiB of CSR storage |

Rows may be empty, and so may a whole matrix. A and B may be rectangular. Two nonempty
inputs may have an empty product.

## Public cases

`cases/hw2-2/<case>/` holds `A.csr`, `B.csr` and the exact product `C.csr`. Hidden cases
follow the same families and add boundaries no public case reaches.

| case | what it is | M | K | N |
|---|---|---:|---:|---:|
| c01 | high collision: many k contribute to one C[i,j] | 7 | 97 | 11 |
| c02 | empty inputs | 3 | 5 | 7 |
| c03 | A nonempty, B empty | 5 | 4 | 9 |
| c04 | a product landing in the last column | 3 | 5 | 8192 |
| c05 | one full A row, the largest output value | 1 | 131072 | 1 |
| c06 | one full B row | 1 | 1 | 8192 |
| c07 | K at its maximum, nothing stored | 1 | 4194304 | 1 |
| c08 | M at its maximum, nothing to multiply | 8192 | 1 | 1 |
| c09-c12 | small random | | | |
| c13 | a scalar | 1 | 1 | 1 |
| c14 | nonempty inputs, empty product | 3 | 5 | 8192 |
| p01 | mixed heavy and light rows | 4093 | 2097169 | 6143 |
| p02 | the same with B's work skewed onto few owners | 4093 | 2097169 | 6143 |
| p03 | ordinary rows clustered in few owners | 6143 | 2097169 | 6143 |

## Rules

- No threads of any kind (OpenMP, `std::thread`, pthreads, `std::async`, parallel STL), no
  GPUs, no third-party libraries. C++17, its standard library, MPI, MPI shared-memory windows
  and compiler SIMD intrinsics are allowed.
- Use only the shards the driver hands you: do not reopen the input files, load precomputed
  answers, or move input-dependent work outside `multiply()`.

Breaking a rule scores 0 on the problem. The judge reads your source and the binary's imports.
