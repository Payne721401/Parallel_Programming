# CUDA Graph submission experiment

Status: implementation and local checks only; NVCC/H100 correctness and speed
must be established on Nano5. Default remains `PP_GRAPH=0` (also when unset).
Temporal blocking is a separate future experiment; no update kernels are fused here.

## Implementation and invariants

`PP_GRAPH=1` records 32 generations on a private nonblocking stream, instantiates
the graph once per `run()`, and replays complete batches. Fewer than 32 iterations
use normal launches. An initially empty sparse worklist still does no updates.
An even batch returns both grid pointers and queue indices to their original
parity. Tail generations use the same stream and update the final pointer normally.
Capture records commands without executing them; no warm-up advances the state.

Sparse preparation increments a device epoch stored in the fourth counter slot.
Every update reads that epoch after its predecessor prepare kernel. Capturing a
host `t+1` alone would reuse stamp values on replay. The ordinary path still uses
the host epoch and passes a null device-epoch pointer. All counters reset per run.
The dense fallback flag remains device-resident and can change during replay.
Counts, graph batches and dispatch depend on input properties, never case IDs.

Default-stream initialization completes once before capture. The private stream
finishes once after all graph replays and tail launches, before default-stream
profiling markers and blocking D2H. There is no per-generation host wait.
Graph objects and stream are destroyed before `run()` returns. Errors are fatal
with an operation label; this candidate does not silently retry after partial work.

Capture, instantiation, first graph launch/upload, waits, tail and destruction all
count toward official Compute time. `[PP_GRAPH]` is printed only with PP_PROFILE=1;
`build_host_ms` covers stream creation/capture/instantiation, not first-launch upload.
It is nested in `kernels_ms`, which still measures the whole interval including
CPU submission/build gaps. Destruction is in `free_host_ms`. Do not add nested times.

## Correctness and bounded A/B

Compile on the login node. Reuse the completed feature matrix inputs and goldens;
the runner does not regenerate or overwrite them. No Nsight Systems is needed.

```bash
make hw3-1 tests/test_hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 5 bash tests/run_graphs.sh \
  tmp/features-405871-20261024-234753-740653 compare
```

Four gates run 38 CPU-reference cases each: graph off/forced sparse, graph
on/dense, graph on/forced sparse, graph on/forced sparse SAT. They cover T=0/1,
31/32/33, 63/64/65, 97 and 5000; odd widths, aligned widths, moving gliders,
torus seams, dead boundaries, decay, empty support, growth into dense fallback,
Moore/VN, M1, C4/C16, direct R2, packed R1 and tiled R5. The harness compares
every cell. Existing feature benchmarks explicitly disable PP_GRAPH.

Then run graph off/on/on/off on six workloads: r1_short, r1_long, r1_c4,
r1_large, forced-sparse r1_cluster, and moore_r5. Total 24 measured executions,
two samples per method, all checked against retained goldens. The old full-size
goldens are a GPU SAT cross-check; the small gates provide independent CPU checks.
No synthetic time limits are invented. Each process has a safety timeout.

Outputs: session.log, samples.tsv, summary.tsv, and individual run logs under
`tmp/graphs-<job>-<timestamp>-<pid>/`. In `compare` mode, profiling is enabled
equally for both methods. Graph dispatch/replays/tail and unchanged sparse work
statistics are validated. `off_over_on > 1` means a measured improvement.
If ranges are close or there are allocation outliers, do not promote by mean alone.

An optional second run with `ab` instead of `compare` disables profiling for the
timed processes, retaining correctness gates. Do it only if the first experiment
suggests a meaningful benefit. New defaults and a full judge are separate decisions.

## Evidence motivating this experiment

User-provided job 406000, revision cab87e7, H100, Nsight Systems 2023.4.4:
dense r1_large had 256 updates totaling 28.147 ms. Sparse r1_cluster had 256
updates totaling 2.422 ms plus 256 prepares totaling 0.384 ms. Earlier matrix
Compute means were 31.605 ms and 4.780 ms respectively. These are different runs;
their difference is not a precise launch-gap measurement. CPU CUDA API duration
overlaps GPU work. No graph speedup has yet been measured.

The user-supplied leaderboard reports roughly 0.2-0.3 s totals, versus the prior
2.573 s full public run (before sliding promotion). This motivates broader
algorithm/data movement work; it does not establish competitors' implementations
or promise that Graphs can close the gap. Public 09's prior full H2D+D2H was about
40 ms, already comparable to the supplied leading total time for that case.

API reference: https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__GRAPH.html
