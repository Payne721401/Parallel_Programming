# HW3-1 baseline and validation

## Status

The working tree now promotes tiled Moore R>=5 to sliding under PP_MOORE=auto
(also the unset default); PP_MOORE=sat retains the old implementation. This new
default still needs a full public integration run. The feature matrix/trace tools
are implemented but not yet executed on Nano5: see [FEATURE_MATRIX.md](FEATURE_MATRIX.md).
Revision 152fec2 compiled on
Nano5 and passed 78 CPU-reference checks. User-supplied job 405529 measured
SAT/sliding speedups of 1.424x on 05 and 1.216x on 09 in a profiled ABBA run.
The subsequent unprofiled R20 case 07 ABBA passed all four runs: SAT mean
0.189035 s, sliding mean 0.1351475 s (1.399x, 28.5% time reduction).
Unprofiled 05/09 and hidden performance remain unmeasured.
See [SLIDING.md](SLIDING.md) for the evidence and next targeted check.

Revision 0713fba adds dense packed R1 and compiled on Nano5. User-supplied job
404889 passed 90 default dense and 30 sparse GPU/CPU-reference comparisons,
then passed the official checker and limits for 11/12 (0.202853 / 0.317 s).
Defaults are R1 packed-byte/shared-memory counting in both schedules, R2-4
direct counting, and R>=5 tiled counting (Moore sliding, von Neumann row prefixes).
See [DENSE_R1.md](DENSE_R1.md)
for the change, limits, and the targeted 11/12 validation command.
Sparse R1 already uses packed-byte/shared-memory counting. Revision a284817
compiled on Nano5, passed 94 small GPU/CPU-reference comparisons, and passed
the official checker and published limits for 08/10 in job 402239.
`PP_R1=direct` restores direct R1 counting in either schedule; `PP_R1=packed`
requests packed R1 unless PP_KERNEL=tiled. Other radii are unaffected.
`PP_KERNEL=auto|naive|tiled` controls counting independently
of `PP_ACTIVE=auto|0|1`, which controls scheduling. Both defaults are `auto`.
The older direct sparse version passed single-case judge 09 in 0.571 s.
The latest packed version passed 08 in 0.426054 s and 10 in 0.390084 s with
profiling enabled and default dispatch. The test initializer-list fix compiled
successfully. A subsequent user-supplied full public judge run passed 12/12
with total 2.573 s; see below. The judge excerpt does not identify its revision
or job ID. Hidden correctness/performance remains unverified.
No graph, mapped-host, or one-bit-per-cell path is enabled. The initial copy,
allocations, launches, final copy, and frees are inside `run()` and count toward
the official Compute time. T=0 returns without allocating anything.
Optional `PP_PROFILE=1` diagnostics report phase times to stderr; unset or 0
disables all profiling events, clock reads and additional synchronization.

This workstation has no detected nvcc/CUDA toolkit. User-provided Nano5 logs
confirm that revision 1cd33f4 compiled, passed 22 tiled reference comparisons,
and passed the official checker on 05-07 with profiling enabled. User-provided
logs also confirm c102594 built and passed single-case judge 05 in 0.290 s
with PP_KERNEL/PP_PROFILE unset (4 GPU seconds held). That verifies the previous
default-dispatch fix, not the current sparse changes. No same-allocation
A/B result has been supplied; do not treat historical ratios as precise speedups.
The original reference and the fixed main.cu/ca.hpp are unchanged.

## Full public judge result (2026-10-24)

User supplied the following result after the dense R1 validation. Build reported
hw3-1 up to date; the excerpt has no exact revision, job ID or environment dump.
The recommended invocation clears PP_KERNEL, PP_ACTIVE, PP_R1 and PP_PROFILE.

| Case | Verdict | Compute s |
| --- | --- | ---: |
| 01 | AC | 0.003 |
| 02 | AC | 0.003 |
| 03 | AC | 0.003 |
| 04 | AC | 0.010 |
| 05 | AC | 0.224 |
| 06 | AC | 0.215 |
| 07 | AC | 0.189 |
| 08 | AC | 0.423 |
| 09 | AC | 0.573 |
| 10 | AC | 0.474 |
| 11 | AC | 0.180 |
| 12 | AC | 0.275 |

Judge total: 12/12, 2.573 s (displayed per-case values are rounded).
GPU held: 0.39 GPU-minutes, reported as 23 s x 1 GPU.
User-reported leaderboard position: 50/66 at that time, not independently checked.
The old 24.04 total included failed-case penalties; its ratio to 2.573 is not
a measured kernel speedup. Cases 08-10 account for approximately 57% of the
displayed total. Case 09 is nearest its limit: 0.573 / 0.7 s.
Preserve this passing implementation as the baseline before further experiments.
No full sweep needs repeating just to reconfirm this result.

## Dense R1 validation (2026-10-24)

User-supplied log: revision 0713fba4ca2d3558b84582765aac21c3186ebc3e,
job 404889, H100 80GB HBM3, driver 550.127.08, CUDA_MODULE_LOADING unset.
Main and test executables compiled with nvcc 12.4. Default dense gate passed
90/90; forced sparse regression gate passed 30/30 (including R5).
Public cases used default dispatch with PP_PROFILE=1:

| Case | Verdict | Compute s | Limit s | H2D ms | Kernel interval ms | D2H ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 11 | AC | 0.202853 | 0.8 | 4.960000 | 191.184479 | 5.467552 |
| 12 | AC | 0.317000 | 0.8 | 2.523584 | 310.848877 | 2.745536 |

Both report profile_status=ok and profile_kernel=dense_r1. Case 11 reports
sparse_selected=0 and initial_work=tiles_total=131072, confirming rejection
of sparse entry. Its zero sparse work statistics do not mean no computation:
the dense kernel performed the updates. Case 12 does not enter sparse inspection.
Logs: tmp/dense-r1-404889-20261024-174746-1798483/default.

The subsequent full public judge integration run is recorded above. No further
A/B or temporal blocking is needed just to pass 11/12. Hidden performance remains
unverified. This record is based on supplied cluster output, not local CUDA execution.

## R1 profile evidence and packed candidate (2026-10-23)

Job 402205, logs tmp/baseline-20261023-222106-306461, user supplied:

| Case | Compute s | H2D ms | Kernel interval ms | D2H ms | Initial work | Mean work | Max work | Dense steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 08 | 1.268750 | 19.515808 | 1225.694702 | 21.062592 | 4549 | 9230.788462 | 12893 | 0 |
| 10 | 0.994618 | 19.541344 | 951.790039 | 20.806784 | 23487 | 38371.295107 | 54638 | 0 |

Both report sparse_selected=1, profile_kernel=sparse_naive, and 524288 total
tiles. The runner's TLE verdict is assigned only after the official checker
accepts the output; PASS 0/2 and exit code 1 mean both missed the time limits,
not failed output validation. The two profiles completed successfully.

Mean scheduled fractions are 1.761% and 7.319%; the kernel intervals account
for 96.606% and 95.694% of Compute. Those intervals still include discovery,
queue management and launch gaps: they do not isolate arithmetic vs atomics.
Scheduled cell updates over all generations are 68,812,943,360 and
51,394,205,696, including inactive cells within scheduled tiles.

The packed implementation replaces only R1 sparse tile arithmetic with four byte lanes
per uint32_t plus shared-memory staging. Each 64x32 tile loads 18x34 words (one
word halo horizontally, one row vertically). State and alive tables together
use 4896 bytes. Each of 256 threads updates two words/eight cells. Counts never
exceed nine per byte, so sums have no inter-byte carry; comparisons use high-bit
sentinels to prevent inter-byte borrow. General C2..C16 transitions are handled.
Aligned rows use 32-bit memory operations; other widths use scalar packing and
valid-byte masking. Grid storage stays one byte per cell; this is instruction
packing, not a fourfold reduction in grid storage or a guaranteed fourfold speedup.

Scheduling, tile dimensions, worklist atomics, fallback thresholds and full-grid
copies are unchanged. PP_R1=auto selects packed only for sparse R1 with automatic
PP_KERNEL. Explicit PP_KERNEL=naive preserves direct unless PP_R1=packed is set;
PP_KERNEL=tiled takes precedence. Profile reports sparse_r1 for the new path.

Local validation: test_r1_packed.cpp compiled with GCC C++17/O2 and warnings as
errors; 20,079,304 scalar comparisons of the exact production header passed.
These exhaust byte equality, count ranges, C2..C16 transitions and compare
random packed Moore/diamond neighbourhoods with scalar counts. The GPU checks
and timings below provide subsequent hardware validation. The runner has Bash
syntax/Slurm guard checks and its profile label is accepted by the parser.

Command used for the completed packed validation (no routine rerun needed):

```bash
cd ~/Parallel_Programming && git pull && cd hw3
unset PP_KERNEL PP_ACTIVE PP_R1 PP_PROFILE
make hw3-1 tests/test_hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 2 bash tests/run_r1.sh profile
```

This runs 64 R1 boundary/rule comparisons and 30 sparse regressions before
profiling only 08/10 with default dispatch. No public 09 rerun or full judge.
Use the printed session.log; expected candidate label is sparse_r1. Optional
`ab` runs direct/packed/packed/direct without profiling in the same allocation;
do not add this routinely just to repeat the successful candidate check.
Submission now also needs r1_packed.hpp beside tile_geometry.hpp and hw3-1.cu.
Suggested commit: `hw3-1: pack R1 sparse updates into shared-memory byte lanes`.

### Packed GPU validation: job 402239

User supplied revision a284817df59d4253fa71dea5b052e2b2a29a3502, H100 80GB HBM3,
driver 550.127.08, CUDA_MODULE_LOADING unset. Both main and test executables
compiled successfully with nvcc sm_90/C++17. Logs:
`tmp/r1-402239-20261023-224701-1948449` (public TSV under `default/`).

- Gate 1: PASS 64/64 packed R1 aligned/unaligned boundaries and rule cases.
- Gate 2: PASS 30/30 motion, decay, stale cleanup, growth and R5 regressions.
- Public 08/10: default dispatch, PP_PROFILE=1, profile_kernel=sparse_r1.

| Case | Verdict | Compute s | H2D ms | Kernel interval ms | D2H ms | Alloc ms | Free ms | Run host ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 08 | AC | 0.426054 | 19.522911 | 382.049652 | 21.850945 | 1.122966 | 1.411161 | 426.006573 |
| 10 | AC | 0.390084 | 19.531200 | 346.076172 | 21.833023 | 1.175486 | 1.373719 | 390.036835 |

Both limits are 0.7 s. Sparse initial/mean/max work counts exactly match job
402205 above, with zero dense fallback steps. Scheduled work did not decrease;
the observed reduction is in time to process it. Total-time ratios are about
2.98x for 08 and 2.55x for 10 across separate jobs, not controlled same-allocation
speedup measurements. Preserve this validated revision; no further 08/10 quota
is needed just to reconfirm these results. Next unfinished paths are 11/12.

## Sparse candidate (2026-10-23; partial GPU validation)

Implementation walkthrough: [SPARSE.md](SPARSE.md).

User-supplied single-case judge results (separate jobs, no phase profiling):

| Case | Verdict | Compute seconds | Limit seconds | GPU seconds held |
| --- | --- | ---: | ---: | ---: |
| 08 | TLE | 1.333 | 0.7 | 4 |
| 09 | AC | 0.571 | 0.7 | 3 |
| 10 | TLE | 0.989 | 0.7 | 4 |

The judge excerpts did not print a revision or chosen kernel. At that point,
local HEAD was ccd2825 with the test initializer fix uncommitted. These were observed timings,
not a controlled speedup measurement. Case 09 is verified correct within its
limit; 08/10 still required checker confirmation and bottleneck diagnosis then.
Both have R=1 and used direct counting before the packed change. Subsequent
profiles and validation above resolve the previous unknowns.

Historical diagnostic command (completed in job 402205): only 08 and 10, using the already
built main binary (no dependency on the test executable or its pending fix):

```bash
srun -A ACD115218 -p dev -N1 -n1 -c4 --gpus-per-node=1 -t 1 \
  env -u PP_KERNEL -u PP_ACTIVE PP_PROFILE=1 \
  python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 15 \
  cases/08.txt cases/10.txt
```

Run from hw3/. The runner writes a unique tmp/baseline-TIMESTAMP directory,
prints phase/worklist fields, and checks outputs even when compute exceeds the
limit. TLE makes the runner exit 1 after both inputs; a Slurm exit-code message
then need not mean a CUDA failure. Inspect each case verdict and retained log.
Do not rerun 09, the full judge, or the full profile script just for this diagnosis.

- Auto inspects initial state only for >=2^28 cells and >=128 generations.
  A GPU scan marks nonzero tiles; a second kernel constructs the radius-dilated
  initial worklist. One scalar D2H selects sparse entry only if work <=25% of
  all tiles. These are heuristics, not measured crossover thresholds.
- `PP_ACTIVE=0` bypasses all sparse discovery/metadata. `PP_ACTIVE=1` forces
  sparse entry for small correctness cases too; it still permits dense fallback.
- Each sparse generation resets the next queue in a one-thread kernel, then
  a worker kernel processes the current list using a block-stride loop.
  Initial worker count is min(tile count, 4*SM count), not a claim of occupancy.
  There are no per-generation host count transfers or device synchronizations.
- Input/output block OR reductions include decay states. Output support dilates
  into the next list; input support adds its own tile for stale-buffer cleanup.
  Atomic epoch stamps deduplicate list entries; each list has capacity for every
  tile, so unique appends cannot overflow. Kernel completion publishes IDs.
- If work exceeds 50%, GPU switches permanently to full-grid tile processing,
  avoiding further flag reductions and queue atomics. It does not re-detect
  sparsity after that switch. Forced entry retains this safeguard.
- Initial dst is zeroed once; no full-grid memset occurs per generation.
  A provably empty initial grid skips all evolution launches. Final D2H remains
  a full-grid copy. Include `tile_geometry.hpp` and `r1_packed.hpp` beside
  `hw3-1.cu` in submission.

Profile additions (`PP_PROFILE=1`):

- `profile_kernel=sparse_naive|sparse_tiled` means sparse entry was selected;
  inspect `sparse_dense_steps` to see whether it later switched to dense.
- `sparse_initial_work`, `sparse_mean_work`, `sparse_max_work` count scheduled
  tiles (including halo expansion and cleanup), not live cells.
- `sparse_tiles_total` is the denominator for work ratios. `sparse_selected=0`
  means initial-density fallback; mean/max work fields are then unmeasured zeros.
- `kernels_ms` includes initial GPU discovery, its scalar-transfer/host-decision
  interval, clearing metadata/dst, queue preparation, and evolution. It is not
  pure arithmetic time. `d2h_ms` also includes the optional tiny statistics copy.

Completed locally (no GPU): CPU geometry test compiled with GCC C++17/O2 and
warnings-as-errors; 103,570 exact axis-influence comparisons and 24 full-grid
sparse buffer/list models passed. Both modified shell runners passed Bash
syntax checks; Python syntax and 10 profile/sparse parser checks passed. The
sparse runner correctly refused execution outside Slurm. These do not validate
CUDA compilation, collectives, or atomics.

After committing/pushing all changes, including the new header, run on Nano5:

```bash
cd ~/Parallel_Programming && git pull && cd hw3
unset PP_KERNEL PP_ACTIVE PP_PROFILE
make hw3-1 tests/test_hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 2 bash tests/run_sparse.sh profile
```

Two minutes is the allocation limit; the script exits as soon as work finishes.
It runs 22 dense, 22 forced-entry, and 30 targeted sparse reference comparisons
before public 08/09/10 with the real defaults. Any wrong answer, runtime error,
or timeout stops the script; correct-but-slow TLE continues to collect the other
cases. No full-size naive A/B and no full judge sweep are run. Use `gate` instead
of `profile` to run correctness only, or `check` for unprofiled public timing.
Do not run all three modes routinely; start with the single profile run above.

Inspect only this run's logs (replace RUN with the printed directory):

```bash
cat RUN/session.log
column -t -s $'\t' RUN/08/results.tsv
column -t -s $'\t' RUN/09/results.tsv
column -t -s $'\t' RUN/10/results.tsv
```

Suggested commit message: `hw3-1: add adaptive sparse tile scheduling and validation`

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
Compute times for each measurement. Require checker success for candidates;
use alternating A/B runs in the same allocation to establish performance
differences. The current default integration fixes the unselected validated
path; R=5 is a conservative heuristic, not a measured optimal crossover. The in-memory
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

`PP_KERNEL=naive` preserves the baseline; `PP_KERNEL=tiled` forces tiled counting
for either shape and all radii. Unset or `auto` selects tiled for R>=5 and naive
otherwise. Invalid values are rejected. The choice is read once in run(), not
once per generation. The initial candidate was opt-in; its successful user
validation and subsequent default-routing correction are recorded below.

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
and 22 candidate comparisons were later confirmed by the user's Nano5 log;
none has run locally.

After committing/pushing, on Nano5:

```bash
cd ~/Parallel_Programming && git pull && cd hw3
make hw3-1 tests/test_hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 --gpus-per-node=1 -t 1 bash tests/run_tiled.sh
```

This now unsets PP_KERNEL to test the actual judge default, rather than forcing
the candidate. It runs 22 small byte-for-byte comparisons in one process (20-second safety
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
still does not establish an optimal radius threshold or performance for all other cases.

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
The user has confirmed GPU compilation/execution of this diagnostic path at
revision 1cd33f4 (recorded below); it cannot be compiled on this workstation.
Local tooling validation passed: ten profile-parser boundary checks and four
real subprocess/log/TSV round trips with synthetic diagnostics (on, off,
malformed, and wrong-answer cases), plus Python/Bash syntax and the Slurm guard.
These tests verify parsing, units, and verdict preservation, not GPU timings.

Suggested commit message when committing the tiled candidate and diagnostics together:

```text
hw3-1: add tiled prefix sums and optional phase timing
```

## Validated tiled run and judge default correction (2026-10-22)

User-provided evidence: job 399745, revision
1cd33f40759fb01099e5334bd3af9b720c69d970, H100 80GB HBM3, driver 550.127.08,
CUDA_MODULE_LOADING unset. Nano5 log directory:
`tmp/tiled-399745-20261022-175224-3500648/`.
The forced tiled path passed all 22 quick comparisons. Public results:

| Case | Checker/verdict | Compute s | Alloc ms | H2D ms | Kernels ms | D2H ms | Free ms | Run host ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 05 | AC | 0.231243 | 0.433978 | 3.820352 | 213.022400 | 3.775072 | 10.104946 | 231.207857 |
| 06 | AC | 0.214011 | 0.454332 | 4.295392 | 203.791397 | 4.279840 | 1.100376 | 213.973814 |
| 07 | AC | 0.200238 | 7.207572 | 4.333024 | 177.488358 | 4.318688 | 6.796393 | 200.196723 |

A subsequent plain judge run still reported 05=1.721 TLE, 06=1.940 TLE,
07=5.508 TLE+, and 4/12 overall, holding the GPU for 47 seconds. Inspection
confirmed that the source still defaulted to naive when PP_KERNEL was unset;
the prior test script had forced PP_KERNEL=tiled. This explains the discrepancy
without evidence of a tiled-kernel regression. The judge log itself did not
print the selected kernel/environment, so that invocation's exact environment
is not independently verified.

Correction: the source now defaults to auto (R>=5 tiled, otherwise naive),
while explicit naive/tiled overrides remain available. The short check/profile
script now clears PP_KERNEL and exercises default dispatch. The harness labels
an unset mode as auto, and profiling continues to report the actual chosen kernel.
This selection is based on radius, never on case number, size/seed fingerprints,
or precomputed outputs. It also routes case 09 through tiled; 09 has not been
validated and still lacks sparse scheduling. No claim is made that 08-12 now
meet their limits. Small-radius cases 01-04 and 08/10-12 keep the baseline path.

After committing/pushing the correction, a single-case judge check avoids
another full sweep over known unfinished paths:

```bash
cd ~/Parallel_Programming && git pull && cd hw3
unset PP_KERNEL PP_PROFILE
make hw3-1 && hw3-1-judge 05
```

Suggested commit message:

```text
hw3-1: select tiled counting automatically for large radii
```
