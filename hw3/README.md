# HW 3-1: a cellular automaton in CUDA

You get a working, sequential program that runs a two-dimensional cellular automaton for T
generations on one CPU core. Your job is to move it onto one H100 and make it fast. The spec
defines the automaton; `ca.hpp` states the same rule in a page.

## Files

| file | status | what it is |
|---|---|---|
| `main.cu` | fixed | `main()`: reads the case, builds the first generation, calls your `run()`, writes the output, prints the time |
| `ca.hpp` | fixed | the rule and `run()`'s contract: **read this first** |
| `hw3-1.cu` | **yours** | `run()` and everything it calls. The only file you submit. |
| `Makefile` | for you | builds the same thing the judge builds |
| `examples/` | for you | small cases with their expected outputs |

**Fixed** means the judge does not read your copies: it compiles your `hw3-1.cu` against its own
`main.cu` and `ca.hpp`, with the flags in the `Makefile`, and runs that. Headers of your own
(`.cuh`, `.hpp`, `.h`) may sit beside `hw3-1.cu`; the judge compiles with your directory on the
include path.

## Build and run

```sh
module load cuda/12.4
make                                         # -> ./hw3-1, the same build the judge does
srun -A <project> -p dev -n1 -c4 --gpus-per-node=1 ./hw3-1 cases/01.txt out.txt
hw3-1-check cases/01.golden.txt out.txt      # compare with the reference output
```

The judge's build is `nvcc -std=c++17 -O3 -arch=sm_90 main.cu hw3-1.cu`; the flags cannot be
changed. `module load cuda/12.4` puts `nvcc`, `nsys` and `ncu` on your PATH; the `Makefile` names
the judge's CUDA 12.4 and host compiler by path, so it builds the same either way. The judge runs each case on one H100 with four CPU cores, one case at a time.

The time that counts is the `Compute time` `main()` prints: from the call to `run()`, with the
first generation already in pinned host memory, until `run()` has returned and the device is
synchronized. Reading the case, creating the CUDA context, building the first generation and
writing the output are not on it; everything `run()` does is.

The public cases and their reference outputs are in `cases/`. The hidden cases are of the same
kinds, with different sizes, rules and seeds.

GPU time on Nano5 is billed to your own wallet, by the minute. Debug on small cases, and cancel
any job you are not using.

## Check correctness on small cases first

The provided `hw3-1.cu` takes hours on the larger cases. Use it as the reference on small
inputs only. `examples/` holds small cases with their expected outputs, among them the patterns
of the spec's first figure:

```sh
srun -A <project> -p dev -n1 -c4 --gpus-per-node=1 ./hw3-1 examples/s2.txt out.txt
hw3-1-check examples/s2.out out.txt
hw3-1-viz examples/s2.txt                    # every generation of a small case, as a GIF
```

Any input of your own in the same format is a test case too: the sequential program gives its
expected output.

## Input and output

The input is a text file: `W H`, the rule string, the boundary (`torus` or `dead`), `T`, then
`n` seed rectangles `x y w h p seed`. `main()` reads it and builds
the first generation: dead except inside the seed rectangles, where each cell is alive with
probability `p` percent. It hands `run()` that grid and the rule. `run()` leaves the last
generation in the same memory, and `main()` writes from it: a hash of the whole grid and the
number of cells in each state.

## Rules

- **The CUDA runtime only.** No Thrust, CUTLASS, CuTe or any other library that does the
  computation for you, and no loading one at run time. CUB, cooperative groups and inline PTX are
  allowed. The judge searches your source for these.
- **The GPU does the computation.** Every generation is computed on the GPU. Host code may
  allocate memory, copy data, launch kernels and decide what to launch, but compute no part of
  a generation. No host threads, OpenMP or parallel algorithms on the host. TAs read the code.
- **Read only what `main()` gives you.** No files, no precomputed answers, no case detection.
  Choosing a method according to the input is allowed; recognising a particular case is not.

## Correctness

The rule is exact, so the output is too: a case passes when its output is identical to the
reference. `hw3-1-check` makes the judge's comparison: it prints which states have the wrong
number of cells, or that every count is right but the hash is not, and exits non-zero when
the output is wrong.
