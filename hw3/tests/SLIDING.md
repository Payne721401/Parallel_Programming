# Moore sliding-window experiment

## Evidence boundary

Closeout: the working tree now defaults to sliding for tiled Moore R>=5 via
PP_MOORE=auto (or unset), retaining PP_MOORE=sat for fallback. This policy change
is pending full public integration validation; the evidence below predates it.
The candidate now has measured improvements on 05/09 and passed 78 reference
checks on Nano5 (user-supplied evidence below). A subsequent unprofiled ABBA
also improved case 07 by 1.399x. Unprofiled 05/09 and large hidden workload
performance are not yet measured. This workstation has no nvcc/GPU.
Do not infer an atomic or bandwidth bottleneck from aggregate kernel times.

## Profiled ABBA result (2026-10-24)

Revision: 152fec2d2a3e722f15a821e7c1ebdcfd267f4441; job 405529.
H100 80GB HBM3, driver 550.127.08, CUDA_MODULE_LOADING unset.
Both executables compiled with nvcc 12.4. Gates passed 48/48 dense and 30/30
sparse/cleanup/growth/von-Neumann reference comparisons. All eight public runs
passed the official checker and time limits, with automatic scheduling and
PP_PROFILE=1 for both methods. Order: SAT / sliding / sliding / SAT.

| Case | SAT mean s | Sliding mean s | SAT/sliding | Time reduction | SAT kernel ms | Sliding kernel ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 05 | 0.223070 | 0.156627 | 1.424x | 29.8% | 214.473 | 148.059 |
| 09 | 0.572462 | 0.470870 | 1.216x | 17.7% | 529.901 | 428.342 |

Ranges: 05 SAT 0.222764..0.223376, sliding 0.156624..0.156630;
09 SAT 0.572215..0.572709, sliding 0.470537..0.471203.
These observed differences greatly exceed the two-sample within-method spread,
but do not establish a distribution of performance across independent jobs.

Case 09 has identical activity statistics in all four runs:
selected=1, tiles_total=524288, initial_work=6742, mean_work=7997.5795,
max_work=9669, dense_steps=0. Mean scheduled tiles are about 1.525% of the grid.
This is scheduled tile coverage, not live-cell density. The matching statistics
and checker outputs support a faster counting implementation with unchanged
work coverage, rather than a speedup caused by skipped updates. H2D (~19.6 ms)
and D2H (~20.5 ms) are stable. The roughly 101.6 ms reduction for 09 is almost
entirely in the kernel interval; that interval still includes scheduler work.

Logs: tmp/sliding-405529-20261024-220820-12439, rounds 1-sat, 2-sliding,
3-sliding, 4-sat. This evidence is the user's supplied log, not local GPU output.
No default was changed during this ABBA. The R20 follow-up is recorded below; these results do
not establish performance on every hidden rule. Further heavy profiling is not needed
just to confirm that this candidate improved the measured 05/09 runs.

## Unprofiled R20 follow-up (2026-10-24)

User ran case 07 only, in one srun allocation, SAT/sliding/sliding/SAT with
PP_PROFILE=0 and automatic scheduling. All four outputs passed the official
checker and 0.5 s limit. The excerpt contains no job ID or exact revision.

| Method | Sample 1 s | Sample 2 s | Mean s |
| --- | ---: | ---: | ---: |
| SAT | 0.189018 | 0.189052 | 0.189035 |
| Sliding | 0.135201 | 0.135094 | 0.1351475 |

Observed speedup 1.399x; time reduction 28.5%. Logs under tmp/:
baseline-20261024-221214-099234, baseline-20261024-221216-227557,
baseline-20261024-221217-956467, baseline-20261024-221219-632277.
This verifies the R20 public workload without profiling, not all R in 1..32.
The current automatic dispatcher with PP_MOORE=sliding changes public 05/07/09
only: R1 still uses packed, R2 remains direct, and von Neumann remains unchanged.
No need to A/B each unaffected public case; do one full integration run when
a final default policy changes. Keep synthetic parameter/distribution checks
separate from public case IDs. Closeout now promotes the general R>=5 Moore policy;
see FEATURE_MATRIX.md for the next non-public diagnostic workload set.

## Change and tradeoff

`PP_MOORE=auto|sat|sliding` controls only tiled Moore counting. Unset means `auto`,
which uses sliding for R>=5 and SAT for smaller radii explicitly forced to tiled.
`PP_KERNEL=naive` is still direct; default R1 is still packed; von Neumann keeps
its row-prefix algorithm. For small-radius sliding correctness tests, explicitly
set `PP_KERNEL=tiled`. No case ID, seed or expected output is inspected.

The experimental `update_tiled<true, Track, true>` shares the existing row-prefix
construction, then forms horizontal window sums in a separate shared table:

```
H(x,y) = sum(alive(x+dx,y), dx=-R..R)
N(x,y) = sum(H(x,y+dy), dy=-R..R)
N(x,y+1) = N(x,y) - H(x,y-R) + H(x,y+R+1)
```

Four groups of 64 threads each own eight consecutive output rows. Each thread
initializes one vertical sum, keeps it in a register and rolls it down its eight
rows. Warps access adjacent columns. The M0 centre subtraction is applied to a
copy of the window, never to the running sum. Cells still evolve from a separate
global source buffer; this is spatial reuse within ONE generation, not temporal
blocking across generations.

Row-prefix and horizontal tables cannot alias while parallel threads read prefix
endpoints. A block barrier follows each construction phase. All threads reach
the sparse flag reductions, even for partial tiles. The existing sparse loop
barrier protects shared storage before a worker reuses it for another tile.
Nonzero/decay tracking, stale-buffer cleanup, epoch deduplication, queue geometry,
worker count, dense fallback and global buffer swaps are unchanged.

Potential gain: removes the vertical prefix scan and reuses neighbouring output
counts in registers. Cost: builds an extra table, repeats the initial vertical
sum for four row segments, and uses more shared memory. For 64x32 output tiles:

| R | SAT shared bytes | Sliding shared bytes |
| --- | ---: | ---: |
| 5 | 6450 | 11826 |
| 20 | 15330 | 24546 |
| 32 | 25026 | 37314 |

All fit the 48 KiB default per-block budget, but residency/occupancy can still
fall. A source-level count of fewer scans is not a performance result.

## Initial measurement procedure (already passed; do not repeat by default)

Compile on the login node; run only inside Slurm:

```bash
make hw3-1 tests/test_hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 5 bash tests/run_sliding.sh compare
```

The 5-minute limit is a safety ceiling, not a wait: the job exits when the script
finishes. Gates are bounded at 40 seconds each. The script does:

1. 48 sliding dense comparisons against the original CPU reference, covering
   R=1/2/5/12/20/32, C2/4/16, M0/M1, dead/torus, minimal/partial dimensions.
2. 30 forced sparse comparisons, including motion, decay, cleanup, growth into
   dense fallback, R5 partial seams and von Neumann regression.
3. Only public 05 and 09, in SAT / sliding / sliding / SAT order in one allocation.
   Both methods use PP_PROFILE=1, automatic scheduling and the official checker.
4. A summary of mean Compute time, the two-sample range and kernel interval;
   it verifies profile labels and identical sparse work statistics across runs.

Logs: `tmp/sliding-<job>-<timestamp>-<pid>/`, with `session.log` and one
`results.tsv` per round. Expected labels: SAT `tiled`/`sparse_tiled`, candidate
`sliding`/`sparse_sliding`. A dense fallback remains labelled `sparse_*` because
it still executes inside the sparse worker. Inspect `sparse_dense_steps` too.

Correct-but-slow TLE results are retained for analysis. WA/RE/timeout stops the
script. A final correctness message does not imply the candidate is faster.
Two samples per method are screening evidence; small differences may be noise.
`kernels_ms` spans setup, prepare kernels, update kernels and submission gaps,
not just counting arithmetic or atomics. Profiling also adds sparse statistics.

The explicit A/B runner still selects each method itself. After closeout changes
the default, use one full public integration judge, not repeated A/B sweeps.

## Optional alternatives, not extra required runs

To inspect the validated baseline first, without running the candidate or gates:

```bash
make hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 1 bash tests/run_sliding.sh baseline
```

This runs 05/09 once with SAT and profiling. It establishes phase/activity data,
but cannot establish a sliding speedup. Normally `compare` already provides it.

`bash tests/run_sliding.sh ab` runs the same gates and ABBA with profiling OFF.
Use it only when the profiled comparison indicates a useful improvement and a
confirmation without instrumentation is warranted, not automatically every time.

R1/sparse targeted scripts clear PP_MOORE to use the current default. The legacy
run_tiled.sh explicitly sets PP_MOORE=sat to preserve its baseline purpose.
For a normal judge invocation:

```bash
unset PP_KERNEL PP_ACTIVE PP_R1 PP_PROFILE PP_MOORE
hw3-1-judge
```

## Decisions after measurement

- Faster on 05/07/09 (now observed): closeout promotes the default as described
  above, pending an integration run. Small reference checks cover multiple
  radii; large hidden workload performance is still unknown.
- Faster only on 05: inspect sparse scheduling, initial work and fallback in 09;
  this alone does not prove atomic contention.
- No clear improvement: retain SAT. A short Nsight Systems trace can separate
  prepare/update duration and launch gaps; use targeted kernel metrics only if
  a concrete remaining question warrants them. Do not run an unrestricted full
  Nsight Compute capture across thousands of generations.

No default is promoted automatically by the runner or summary.
