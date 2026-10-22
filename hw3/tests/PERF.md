# HW3-1 baseline and validation

## Status

The first CUDA implementation uses one thread per cell, a 32 x 8 thread block,
two byte grids on the GPU, and one kernel launch per generation. No tiled,
sparse, graph, mapped-host, or bit-packed path is enabled. The initial copy,
allocations, launches, final copy, and frees are inside `run()` and count toward
the official Compute time. T=0 returns without allocating anything.

This workstation has no detected nvcc/CUDA toolkit. GPU compilation, full-grid
comparison, official checker results, and H100 timings remain **unverified**.
Do not infer performance from the baseline architecture or from CPU tests.
The original reference and the fixed main.cu/ca.hpp are unchanged.

Local checks completed on 2026-10-22:

- Python syntax checks and parsing all six provided example descriptions.
- Runner exercised with real temporary subprocesses: AC, WA, TLE, RE,
  WALL_TIMEOUT, missing output, and missing/invalid timing classifications.
- Mocked official-checker failure and timeout handling; old logs cannot be overwritten.
- Bash syntax check and rejection of execution without SLURM_JOB_ID.
- Git diff whitespace check and LF line endings for the added scripts.

These checks validate test tooling, not CUDA compilation or GPU results.

## First Nano5 run

After committing and pushing the local change, on the Nano5 login node:

```bash
cd ~/Parallel_Programming && git pull && cd hw3
make hw3-1 tests/test_hw3-1
wallet ACD115218
srun -A ACD115218 -p dev -N1 -n1 -c4 --gpus-per-node=1 -t 2 bash tests/run_baseline.sh
wallet ACD115218
```

Set up the course checker once if it is not already on PATH:

```bash
/work/dogeon188/pp26/judge/bin/pp26-setup
source ~/.bashrc
```

The script checks 206 in-memory cases against the unmodified CPU reference,
comparing every byte, in one process/context. It covers rule modes, T=0/1 and
odd/even T, minimal and partial blocks, R=16/31/32 on small grids, empty grids,
and decay states. Arbitrary initial decay states deliberately test the run()
contract more broadly than the provided input generator. A mismatch reports
the seed (at startup), case index, rule, dimensions and first mismatching cell.
The default PRNG seed is fixed; `./tests/test_hw3-1 64 NEW_SEED` selects another.

The six official examples then run through the fixed main and official checker.
Expected success is `PASS 206/206`, followed by `PASS 6/6`; these are expectations,
not results already obtained. Logs are under `hw3/tmp/baseline-JOBID-*/`:

- `session.log`: revision, GPU model, test output and errors.
- `case-stats.tsv`: public case parameters, when the cases link is available.
- `examples/results.tsv`: verdict, Compute time, whole-process time, time limit.
- `examples/*.log`, `*.check.log`, `*.out`: per-case evidence.

The allocation has a two-minute cap (about 1.67 SU at 50 SU/hour). It exits as
soon as the script completes; queue time does not consume the GPU allocation.
All test execution, including the CPU reference, is inside Slurm. Do not run
test scripts on the login node. Code editing and compilation can happen there.

## Public cases and measurements

The public `cases` link is absent from this local checkout. On Nano5, use the
course-provided cases link. Do not run all public cases with the naive kernel
until inspecting their sizes, radii, and generation counts. The stats script
reads descriptions without materializing grids. `naive_checks` is the nominal
W*H*T*neighbourhood size, not a timing estimate; dead boundaries reduce work.
The rectangle area/probability sums can double-count overlaps, so neither is
an exact active-cell or active-tile fraction.

For an explicitly selected case, run inside a short allocation, for example:

```bash
srun -A ACD115218 -p dev -N1 -n1 -c4 --gpus-per-node=1 -t 1 \
  python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 20 cases/01.txt
```

`run_cases.py` defaults to the six examples. It infers published Compute time
limits only for cases/01.txt through cases/12.txt; use `--limit SECONDS` for a
different path. The wall timeout guards the entire process, including input
generation and hashing; it is not the official Compute time limit. `AC` requires
a matching output and, if specified, Compute time within the limit. Without
`--checker`, the script compares all whitespace-separated output tokens.
It never overwrites an old log directory and returns nonzero on any failure.

Record the revision, environment, GPU, allocation ID, correctness and all
Compute times for each measurement. Future candidate/default changes require
checker success and alternating A/B runs in the same allocation. The in-memory
harness is for correctness; its warm context is not an official timing setup.
No EAGER global initializer is included in this first version.

## CUDA concepts used by this version

- A thread computes one output cell. Adjacent x threads read adjacent columns.
- Source and destination are separate: all neighbours come from one generation.
- Same-stream kernel ordering supplies the dependency between generations.
- Swapping two pointers replaces a whole-grid copy each generation.
- Partial blocks check x/y before accessing memory.
- A torus neighbour can wrap once because valid coordinates plus [-R,R] stay
  within one grid extent. A future padded tile needs a different/general map.
- Only state 1 contributes to neighbour counts; states 2..C-1 still advance.

Suggested commit message:

```text
hw3-1: add naive CUDA baseline and bounded correctness tests
```
