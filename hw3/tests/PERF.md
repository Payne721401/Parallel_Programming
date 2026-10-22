# HW3-1 baseline and validation

## Status

The default CUDA implementation uses one thread per cell, a 32 x 8 thread block,
two byte grids on the GPU, and one kernel launch per generation. An experimental
tiled prefix-sum kernel is available via `PP_KERNEL=tiled`; the default remains
`naive` pending validation. No sparse, graph, mapped-host, or bit-packed path is
enabled. The initial copy,
allocations, launches, final copy, and frees are inside `run()` and count toward
the official Compute time. T=0 returns without allocating anything.
Optional `PP_PROFILE=1` diagnostics report phase times to stderr; unset or 0
disables all profiling events, clock reads and additional synchronization.

This workstation has no detected nvcc/CUDA toolkit. The user has reported a
naive judge run (below); candidate GPU compilation, full-grid comparison,
checker results, and timings remain **unverified**. Do not infer candidate
performance from its architecture or from CPU tests.
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

## User-reported baseline and case parameters (2026-10-22)

Source: user-pasted Nano5 judge output and the first 20 lines of each public
case. The remote revision was not printed; the local baseline commit is
408989d. These are historical baseline timings, not same-allocation A/B data.

| Case | W x H | R | C | M | Shape | Boundary | T | Judge | Reported seconds |
|---|---|---:|---:|---:|---|---|---:|---|---:|
| 01 | 2048 x 2048 | 1 | 2 | 0 | Moore | torus | 200 | AC | 0.023 |
| 02 | 2048 x 2048 | 1 | 3 | 0 | Moore | dead | 200 | AC | 0.010 |
| 03 | 2048 x 1536 | 1 | 4 | 0 | Moore | torus | 200 | AC | 0.009 |
| 04 | 1500 x 2500 | 2 | 5 | 0 | von Neumann | dead | 150 | AC | 0.010 |
| 05 | 14300 x 14400 | 5 | 2 | 1 | Moore | torus | 100 | TLE | 1.722 |
| 06 | 17800 x 13100 | 12 | 4 | 0 | von Neumann | dead | 50 | TLE | 1.940 |
| 07 | 15300 x 15350 | 20 | 2 | 1 | Moore | torus | 40 | TLE+ | 5.510 |
| 08 | 32768 x 32768 | 1 | 2 | 0 | Moore | torus | 3640 | TLE+ | 5.710 |
| 09 | 32768 x 32768 | 5 | 2 | 1 | Moore | torus | 2000 | TLE+ | 5.709 |
| 10 | 32768 x 32768 | 1 | 3 | 0 | Moore | torus | 654 | TLE+ | 5.709 |
| 11 | 16384 x 16384 | 1 | 2 | 0 | Moore | torus | 403 | TLE | 1.437 |
| 12 | 16384 x 8192 | 1 | 4 | 0 | Moore | dead | 1190 | TLE | 1.704 |

The run held one GPU for 47 seconds. The displayed total 24.05 includes eight
3-second failure penalties. TLE+ values are not completed runtime measurements;
correctness of timed-out cases has not been established.

Cases 05-07 initially fill the entire grid at 30%, 40%, and 30%, respectively.
Their neighbourhood sizes are 121, 312, and 1681 (including centre only for M1).
These make redundant counting the first target. Cases 11-12 initially fill the
whole grid at 30% and 25%; their later density is unknown.
Only part of the seed lists for 08-10 was supplied. Case 08 has 2838 rectangles;
the shown ones are tiny and clustered. Case 09 has 510 rectangles; the shown
ones are scattered 28 x 28 patches. Case 10 has 41 rectangles, including a
32768 x 1365 band at 40% plus shown small patches. Do not infer the exact global
density, unshown seed sizes, or subsequent density from those partial lists.
Future sparse scheduling must measure actual GPU data and handle dispersed
activity, not recognize case numbers or seeds.

## Tiled candidate: short validation before more judge runs

`PP_KERNEL=naive` preserves the baseline; `PP_KERNEL=tiled` selects the candidate
for either shape and all radii. An invalid nonempty mode is rejected. The
choice is read once in run(), not once per generation. No default switch or
performance claim is made before correctness and controlled timing evidence.

Each 256-thread block produces a 64 x 32 tile. It loads an R-cell halo as
alive flags and builds uint16 row prefixes with warp shuffles. Moore adds a
column scan to make a summed-area table, then uses four reads per rectangle.
Von Neumann retains row prefixes and uses two reads per each of 2R+1 rows.
M0 subtracts the centre only when it is alive. The centre's original byte is
read separately for the state transition. Maximum shared allocation is 25,026
bytes at R=32, and the largest prefix is at most 12,288. Both are tile-size
bounds, not measured resource occupancy or speed.

Every warp lane executes its shuffles, with zero-padding for partial chunks;
all threads reach block barriers before filtering invalid output coordinates.
General modulo handles tiny torus grids whose halo can wrap multiple times.
The quick reference checks exercise these boundaries, chunk carries, partial
and multiple tiles, several radii, odd/even T, and decay. They are a short gate,
not an exhaustive replacement for the full correctness suite.

Candidate-side local checks: Python/Bash syntax, the Slurm guard, ten A/B
result-gate checks (including refusing WA/RE/timeouts), LF script endings,
and integer/shared-memory size bounds for R=1..32 passed. CUDA compilation
and the 22 candidate comparisons still require Nano5; none has run locally.

After committing/pushing, on Nano5:

```bash
cd ~/Parallel_Programming && git pull && cd hw3
make hw3-1 tests/test_hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 --gpus-per-node=1 -t 1 bash tests/run_tiled.sh
```

This runs 22 small byte-for-byte comparisons in one process (20-second safety
timeout), then only cases 05, 06 and 07 with the official checker (8-second
whole-process timeout each). The allocation cap is 1 minute, about 0.83 SU.
It does not run the full judge or the 206-case suite. Inspect
`tmp/tiled-JOBID-*/session.log` and `tiled/results.tsv`. A time-limit failure
still records output correctness through the runner's checker before reporting
TLE. The script returns nonzero on any unsuccessful candidate case.

Only if the candidate passes, an optional 2-minute A/B job runs naive, tiled,
tiled, naive on 05 and 06 in the same allocation:

```bash
srun -A ACD115218 -p dev -N1 -n1 -c4 --gpus-per-node=1 -t 2 bash tests/run_tiled.sh ab
```

The A/B script permits AC/TLE rows (both require a correct output), but stops
on WA, RE, wall timeout, missing timing/output or tool failure. It skips naive
07 because the prior TLE+ makes it an expensive baseline. This comparison
still does not validate an automatic dispatch policy for all other cases.

Suggested commit message:

```text
hw3-1: add opt-in tiled prefix sums for large-radius neighbourhoods
```

## Optional phase timing (PP_PROFILE=1)

To collect the first candidate correctness/timing results with diagnostics in
the same short job, use `profile` instead of the default `check` mode:

```bash
cd ~/Parallel_Programming && git pull && cd hw3
make hw3-1 tests/test_hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 --gpus-per-node=1 -t 1 bash tests/run_tiled.sh profile
```

The script still runs only the 22-case quick gate and public cases 05-07. It
enables profiling for those three public cases, not for the quick gate. It
does not add another public-case pass. Default `check` and `ab` modes explicitly
set PP_PROFILE=0, even if the caller exported PP_PROFILE=1.
For a separately selected case or the naive kernel, set PP_PROFILE=1 alongside
PP_KERNEL in the environment of the program/runner inside srun.

Each successful profiled run emits one `[PP_PROFILE]` line to stderr (never
to the answer file). `run_cases.py` preserves it in the case log, displays the
fields on the terminal, and appends these columns to `results.tsv`:

| Field | Meaning | Clock |
|---|---|---|
| profile_status | ok, absent, or invalid; absent is not a zero measurement | parser |
| profile_kernel | naive, tiled, or none for T=0 | configuration |
| alloc_host_ms | Two device allocations | host steady_clock |
| h2d_ms | Stream interval bracketing the initial H2D copy | CUDA events |
| kernels_ms | Stream interval covering all T generations | CUDA events |
| d2h_ms | Stream interval bracketing the final D2H copy | CUDA events |
| free_host_ms | Two device frees | host steady_clock |
| run_host_ms | Profiled run span, ending before printing the diagnostic | host steady_clock |

All phase durations are **milliseconds**. The existing compute_s and wall_s
columns remain in **seconds**. The line also includes steps and cells for
context. T=0 emits zero phase times without creating events or calling CUDA;
run_host_ms still measures the small host diagnostic path.

Four events are recorded in the same default stream as the copies/kernels.
There is one final event synchronization after D2H, no per-generation events
or synchronizations, and every event/API error is checked. Event intervals
include gaps between operations, including host submission delays and possible
first-launch/module-loading delays; kernels_ms is not a sum of isolated kernel
durations. Copy intervals likewise are not pure hardware-bandwidth benchmarks.

run_host_ms includes event creation/collection/destruction but excludes printing
the profile line and main()'s final synchronization. Official Compute time
includes those as well. Do not force a sum of mixed-clock fields to match it.
The profiling itself adds overhead, especially on short cases; use PP_PROFILE=0
for final performance comparisons/judge results. A killed or failed process may
never emit a profile line; its missing measurements remain blank, not zero.

CUDA timing API reference:
[CUDA 12.4 Event Management](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART__EVENT.html).
GPU compilation/execution of this diagnostic path still requires Nano5.
Local tooling validation passed: ten profile-parser boundary checks and four
real subprocess/log/TSV round trips with synthetic diagnostics (on, off,
malformed, and wrong-answer cases), plus Python/Bash syntax and the Slurm guard.
These tests verify parsing, units, and verdict preservation, not GPU timings.

Suggested commit message when committing the tiled candidate and diagnostics together:

```text
hw3-1: add tiled prefix sums and optional phase timing
```
