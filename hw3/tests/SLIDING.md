# Moore sliding-window experiment

## Evidence boundary

The default remains the public-12/12 passing SAT/packed implementation.
There is no measured speedup for this candidate yet, and this workstation has
no nvcc/GPU. The older 05-07 phase measurements locate most time in the kernel
interval, not specifically in vertical prefix scans. Case 09 has judge totals
but no recorded phase/activity profile. Do not claim an atomic, bandwidth or
scan bottleneck from these totals alone.

## Change and tradeoff

`PP_MOORE=sat|sliding` controls only tiled Moore counting. Unset means `sat`.
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

## Recommended first measurement

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

Do not run the full judge for this experiment: its default still uses SAT.

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

Existing R1/sparse/tiled targeted scripts clear PP_MOORE so an exported candidate
setting cannot contaminate their baseline runs. For a normal judge invocation:

```bash
unset PP_KERNEL PP_ACTIVE PP_R1 PP_PROFILE PP_MOORE
hw3-1-judge
```

## Decisions after measurement

- Faster on 05 and 09 with matching work: extend validation to 07 and non-public
  radius/distribution cases before considering a general default policy.
- Faster only on 05: inspect sparse scheduling, initial work and fallback in 09;
  this alone does not prove atomic contention.
- No clear improvement: retain SAT. A short Nsight Systems trace can separate
  prepare/update duration and launch gaps; use targeted kernel metrics only if
  a concrete remaining question warrants them. Do not run an unrestricted full
  Nsight Compute capture across thousands of generations.

No default is promoted automatically by the runner or summary.
