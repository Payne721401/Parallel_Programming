# Feature matrix and shared-cost measurement

Status: tools implemented locally; Nano5 compilation/execution is pending.
No timings in this document are invented. The matrix is independent of public
case names/seeds and is a diagnostic workload set, not a hidden-test predictor.

## Sliding closeout

`PP_MOORE=auto|sat|sliding` now defaults to auto. In tiled Moore execution,
auto uses sliding for R>=5, based on the measured R5/R20 improvements and small
R1..32 correctness checks. Explicit sat retains the old implementation; explicit
sliding enables the experimental choice even at smaller R when PP_KERNEL=tiled.
R1 packed, R2-4 automatic direct counting, von Neumann counting and sparse
thresholds are unchanged. Unmeasured intermediate radii are not claimed optimal.
One full public integration run is still needed for this changed default.

## Nine inputs, eleven execution variants

Unless noted, dimensions are 2048x2048, boundary is torus, and dense initialization
is deterministic 35% alive using the same seed. R1 uses S2..3/B3, M0. For larger
neighbourhoods, M1 and thresholds are scaled by neighbourhood size (S28%-48%,
B28%-37%, rounded to valid integers). Raw count thresholds necessarily change
when comparing radius/shape; those comparisons are not a pure hardware microbenchmark.

| Input | R / C / shape | T | Question |
| --- | --- | ---: | --- |
| r1_short | 1 / 2 / Moore | 32 | Fixed costs versus short iteration chains |
| r1_long | 1 / 2 / Moore | 256 | Same initial grid/rule, more generations |
| r1_c4 | 1 / 4 / Moore | 256 | Same initial live grid/count rule, decay states added |
| moore_r5 | 5 / 2 / Moore | 128 | Moderate-radius counting |
| moore_r20 | 20 / 2 / Moore | 128 | Halo and larger-radius counting cost |
| diamond_r5 | 5 / 2 / von Neumann | 128 | Neighbourhood geometry and O(R) interval summation |
| r1_cluster | 1 / 2 / Moore | 256 | 256 stable 2x2 blocks gathered in a small region |
| r1_spread | 1 / 2 / Moore | 256 | Same 256 stable blocks spread over the grid |
| r1_large | 1 / 2 / Moore | 256 | 8192x8192 (64 MiB/grid), larger working set |

Cluster/spread have exactly 1024 live cells (0.024414% of the small grid), with
enough separation for all blocks to remain unchanged under this R1 rule. They
isolate support geometry without uncontrolled motion/extinction. They do not
replace growth/decay correctness tests. Cases with random fills may evolve very
differently; their total time must not be interpreted as a pure per-state cost.

Every input runs with default scheduling. Cluster/spread also run with
PP_ACTIVE=1. All these grids are below the 256 MiB auto-inspection threshold:
auto runs are dense even on sparse inputs. Forced sparse measurements explicitly
show the opportunity/cost that auto currently misses. A widely spread pattern
may immediately invoke the existing one-way dense fallback; inspect dense_steps.
No global threshold is silently changed to make the matrix look better.

The small cases may be cache-resident; r1_large adds a larger working set.
Neither size alone proves an HBM-bandwidth bottleneck, and the matrix is not a
replacement for testing the larger public grids before promoting a new algorithm.

## Run the matrix first

Compile on the login node, then run in one bounded allocation:

```bash
make hw3-1 tests/test_hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 3 bash tests/run_features.sh profile
```

The job ends as soon as the script finishes; three minutes is a safety ceiling.
It does not run the public judge. It runs 22 default-dispatch CPU reference checks
and 30 forced-sparse regressions, then generates the nine input descriptions.
For each large input, the retained dense SAT path generates the expected output
on the GPU, once. Candidate outputs are checked with hw3-1-check. This is a GPU
baseline cross-check, not an independent CPU proof at full size. The small gates
provide the independent reference comparisons.

Eleven variants run twice, first forward then reverse, with PP_PROFILE=1. This
screens run-to-run variability with 22 measured processes, not a precise statistical
benchmark. Each reference process is limited to 15 seconds; measured processes
have a 15-second wall timeout and a 30-second wrapper timeout. Any mismatch,
missing profile or unexpected kernel selection stops the run. There are no
invented synthetic time limits: AC here means matching output with valid timing.

Outputs under `tmp/features-<job>-<timestamp>-<pid>/`:

- `manifest.json`, `inputs/`: exact parameters and deterministic rectangles.
- `samples.tsv`: each observation, including profile and sparse counters.
- `summary.tsv`: mean/range, phases, kernel interval per step, scheduled tile
  fraction, and kernel interval per million scheduled output cells.
- Per-process logs and checker output, plus a complete `session.log`.

For dense execution scheduled tile fraction is 1; it is not live-cell density.
For sparse execution it includes cleanup tiles, halo dilation and any fallback.
All matrix dimensions divide 64x32 exactly, so tiles map directly to cell counts.
The per-cell metric still includes schedule/setup costs: it is not an instruction
throughput or HBM measurement. `kernels_ms/T` likewise includes one-time setup
amortized over T. Profiling adds event/statistics overhead. Do not sum run_host_ms
and nested phases, or infer compute-bound versus bandwidth-bound from these alone.

## Optional second step: two focused traces

Use this only after the matrix passes and when separating scheduling costs is
useful. It requires nsys from the cluster CUDA installation (load cuda/12.4 if
needed). Compile the test-only wrapper:

```bash
make tests/trace_hw3-1
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 4 bash tests/run_features.sh trace \
  tmp/features-REPLACE-WITH-THE-ACTUAL-RUN
```

The separate trace executable includes the fixed main without editing it, and
wraps only run() in cudaProfilerStart/Stop with one final sync. No per-generation
events or host syncs are added. The submission code does not include the wrapper.
Nsight collects CUDA only, with CPU sampling/context-switch tracing disabled.
It records r1_large (dense) and r1_cluster (forced sparse); PP_PROFILE=0, so event
profiling and sparse stats are not mixed into the trace. Both outputs are checked
against the same matrix references. Keep the code revision consistent between
matrix and trace, and retain the printed revision/status when comparing them.

The .stats.txt files show update/prepare kernel counts and total durations, CUDA
API durations, and transfers. The .nsys-rep timeline shows gaps and overlap.
CPU API time overlaps GPU execution: do NOT add it to kernel time, or subtract
it from the benchmark total to invent a launch-overhead estimate. Trace timing
is diagnostic and can be perturbed by the profiler; use untraced A/B for speedups.
Each trace has a 60-second timeout; stats export has a 40-second timeout.

Official API/CLI guidance: https://docs.nvidia.com/nsight-systems/UserGuide/index.html

## Decisions supported by the data

- Large working-set update cost dominates: consider temporal reuse, then measure
  rather than assume bandwidth saturation.
- Many short kernels with visible idle gaps: investigate graph batching.
- Cluster benefits but spread does not: reconsider tile granularity and dispatch,
  not the cellular-automaton rules.
- Fixed overhead dominates short runs: avoid expensive setup on those workloads.

After the changed default passes these gates, perform ONE full public integration
judge with PP_KERNEL/PP_ACTIVE/PP_R1/PP_PROFILE/PP_MOORE unset. The judge submits its
own job; do not wrap it in srun. Do not rerun the previous SAT/sliding ABBA by default.
