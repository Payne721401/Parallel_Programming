# HW2-2: Distributed Sparse Matrix Multiplication

HW2-1 and HW2-2 each contribute 50% of the assignment grade. Within HW2-2,
correctness is 40% and performance is 60%.

Compute the complete exact product C = A B using MPI. A is M-by-K and B is K-by-N.
Absent entries are zero. Stored input values are integers in [1,8]; columns in
each input row are strictly increasing, with no duplicates or stored zeros.
Empty rows and completely empty matrices are valid; dimensions are positive.

## Input bounds

| Quantity | Upper bound |
| --- | ---: |
| M | 8,192 |
| K | 4,194,304 |
| N | 8,192 |
| nnz(A) | 8,388,608 |
| nnz(B) | 536,870,912 |
| nnz(C) | 67,108,864 |
| Nonzeros in one A row | 131,072 |
| Nonzeros in one B row | 8,192 |
| Scalar products, sum over A[i,k] of nnz(B[k,:]) | 8,000,000,000 |
| Initial A+B CSR storage on one rank at P=24 | 512 MiB |
| Final C CSR storage on one rank at P=24 | 32 MiB |

CSR storage in the last two bounds counts 8 bytes per entry and 8 bytes per row
offset; application/MPI temporary allocations are additional. These limits hold
simultaneously. The fixed prototype parser accepts a wider superset for debugging;
the judged dataset follows the bounds above.

A given k contributes at most once to C[i,j], so C[i,j] is at most
64 * nnz(A[i,:]) <= 8,388,608. Output values fit uint32_t exactly; the supplied kernel uses uint64_t accumulation.
Work and size arithmetic must accommodate counts above 2^32 (for example, using
uint64_t). Alternative exact accumulator representations are permitted.
No floating-point tolerance or modulo arithmetic is part of the problem.

## MPI interface

Implement only `hw2-2.cpp`:

```cpp
void multiply(const Matrix& a, const Matrix& b, Matrix& c);
```

The fixed driver calls it once on every rank. With P ranks, rank r initially
owns A rows [floor(M*r/P), floor(M*(r+1)/P)) and B rows with the analogous K
bounds. `rows` and `cols` are global dimensions; `first` is the first owned global
row. Local `ptr` begins at zero and offsets local `data`. Empty owners have
`ptr = {0}`. See the supplied `spgemm.hpp` for the exact representation.

Return C with exactly A's original row ownership. Every C row must have sorted,
unique column indices and no stored zeros. Combine all partial contributions
exactly. Finish all nonblocking operations before returning. Redistribution,
collectives and MPI shared-memory techniques within the allocated resources are
permitted. Do not assume a fixed source rank, divisible dimensions, a particular
row ordering or a power-of-two shape.

The supplied `local_kernel.hpp` contains a reusable dense accumulator with a
touched-column list and sorted sparse-row extraction. It is optional; the TA's
comparisons use this same kernel. The MPI starter replicates B and computes its
original A rows; that approach is not a guarantee of high performance or of
fitting all legal resource combinations.

## Execution and timing

The judge runs every case on **24 MPI ranks across two nodes, 12 ranks per
node, one physical core per rank**, with **48 GiB of memory per node** and
**1200 MiB RLIMIT_DATA per rank**. Shared MPI windows are allowed within the
node's allocation. RLIMIT_DATA is not an RSS or per-window size limit. The
reference solution peaks at about 430 MiB per rank.
Use C++17, the standard library, MPI and compiler SIMD intrinsics.
Additional compute threads (including OpenMP threads), GPUs and external sparse-matrix libraries
are not part of this assignment configuration. Small debugging cases may be run
with other rank counts; performance resource guarantees apply to P=24.

Timing begins after fixed input loading and a barrier. It includes all work
inside `multiply`: statistics discovery, symbolic preprocessing, redistribution,
communication, local products, duplicate merging, sorting and restoration of
C ownership. A completion barrier follows; the maximum rank elapsed time is
reported. Fixed validation and output writing are outside the computation timer.
Use only the provided shards as input to multiplication; reopening input files,
loading precomputed answers or bypassing the fixed driver is not permitted.

Only `hw2-2.cpp` is submitted. The grading build must use TA copies of the
driver, interface and supplied kernel. Correctness checks the entire canonical
CSR output, not a checksum or C times a vector. Public correctness cases have a 10-second compute ceiling; public performance
cases p01_large, p02_skew and p03_clustered have a 5-second compute ceiling.
A separate 60-second wall ceiling covers the whole program, input and output
included. The judge stops a program 10 seconds past it.

## Binary CSR format

All fields are little-endian, without padding:

| Field | Type |
| --- | --- |
| Magic | 8 bytes: SPGCSR01 |
| Rows, columns | two uint32_t |
| nnz | uint64_t |
| Row offsets | rows+1 uint64_t, first 0, last nnz |
| Entries | nnz interleaved (uint32_t column, uint32_t value) pairs |

Indices are zero-based. Row offsets are nondecreasing. The exact byte length is
24 + 8*(rows+1) + 8*nnz. The invocation is `hw2-2 A.csr B.csr C.csr`, and
`hw2-2-check golden.csr C.csr` compares two such files and names the first
row offset or entry that differs.

## Workload families

Public and hidden performance inputs mix broad heavy rows, high-reuse B rows
and low-reuse B rows in the same product. Their output columns can overlap.
Heavy work can be concentrated in several owners, and the remaining work may
be spread or clustered. Hidden cases vary dimensions, degrees, seeds and owner
counts within these announced families. No single initial row or inner-index
interval is reserved for a particular role.

Correctness cases cover empty inputs and ownership, rectangular shapes, high
collision, maximum dimensions/row lengths, the maximum output value, nonempty
inputs with zero product, complete dense output, the nnz(A) limit and a total scalar-product count
above 2^32. Full solutions must handle
all of these, regardless of the strategy selected for performance cases.
