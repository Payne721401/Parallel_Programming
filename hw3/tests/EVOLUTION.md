# Exact sampled evolution diagnostics

Status: implemented locally; NVCC/H100 checks pending. Default PP_EVOLVE=0.
No early exit, dynamic dense-to-sparse switch, or temporal blocking is enabled.

## What is measured

PP_EVOLVE=1 scans the entire current grid and its immediate predecessor at
generation 0, 1, 2, 4, 8, 16, 32, 64, multiples of 128, and the final generation.
Duplicate times are removed. PP_EVOLVE_EVERY=1..5000 changes the regular cadence;
default 128 yields at most 48 snapshots at T=5000. This is time sampling, NOT
sampling a subset of cells. Counts at each observed time are exact.

Each snapshot records:

- alive (state 1), nonzero (states 1..C-1), decay (nonzero-alive), and number of
  cells whose COMPLETE state byte differs from the immediately preceding generation.
- nonzero and changed tile counts (64x32, including clipped edge tiles).
- tile occupancy counts: empty, 1..16, 17..128, 129..512, 513..2048 nonzero cells.
  These bins count cells rather than fractions; edge tiles can be smaller.
- influence_tiles: tiles intersecting the R-expanded nonzero-tile support.
  This is conservative TILE-level coverage, not exact cell-level minimal work.
  Rectangular expansion is also a conservative superset for von Neumann rules.
- work_tiles: influence union tiles with nonzero PREVIOUS-generation cells,
  including ping-pong destination cleanup. This matches the existing worklist
  criterion, independently evaluated from a full-grid scan.
- scheduled_next and dense: current queue/fallback state. Before the next prepare
  kernel, a queue exceeding half the grid can still have dense=0; prepare will
  trigger dense fallback. No diagnostic changes that decision.
- fixed=1 only for t>0 when every state byte of G(t) equals G(t-1).

`first_observed_fixed` is an upper bound on the first fixed generation, not an
exact convergence time unless every generation is sampled. -1 means none observed.
Constant alive/nonzero counts do not imply a fixed point. Period-2 or longer
cycles are NOT detected. Nonzero history between samples is not retained, so this
cannot certify a never-touched set for selective D2H.

## Timing and perturbation

The diagnostic scan and reduction run on the GPU; rows are copied to the CPU only
after the evolution completes. No per-snapshot D2H or host wait is introduced.
`scan_ms` includes the row memset, scan and reduction. `update_ms` spans the prior
snapshot's end to this snapshot's start: updates plus CPU submission gaps, excluding
the explicit scan kernels. Initial update_ms is 0. Normalize by the difference in
generation numbers before comparing intervals.

Meta records separately label diagnostic allocation/event setup, accumulated scan
time, final row copy, and GPU resource cleanup. The normal PP_PROFILE kernel
interval includes scans/setup; run_host_ms and official Compute also include
diagnostic reporting. Host/GPU timings overlap and must not be added mechanically.
Scans perturb caches, scheduling and host submission, and diagnostic allocations
can alter later allocation/free costs. Subtracting scans_ms is NOT a corrected
judge prediction. A matched diagnostic-off run is included for context (one sample
each, not a statistical A/B speedup experiment).

PP_EVOLVE=1 and PP_GRAPH=1 together fail explicitly. Test graph speed separately.
With diagnostics off there are no diagnostic GPU allocations/events/scans/copies.
With diagnostics on an initially empty sparse grid still follows the sampling
schedule through T; ordinary execution retains its zero-work fast path.
Diagnostic T=0 measures the initial state; normal T=0 still returns immediately.

## GPU correctness gate and public collection

The independent test executable evolves small inputs with the original CPU
reference and emits expected counters. A second pass runs production CUDA and
compares final grids. The parser compares every sampled counter to the CPU oracle.
20 cases include dead/torus boundaries, empty T=0/T=33, stable blocks, a period-2
oscillator, moving gliders, C16 decay, C4 dense state, R5 Moore/VN and sparse growth.
Dense mode tests the default cadence; forced sparse tests every generation.
The CPU oracle scatters occupied-tile influence while the GPU gathers neighbours.

Compile on the login node, then use one allocation:

```bash
make hw3-1 tests/test_evolution &&
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 5 bash tests/run_evolution.sh
```

Default inputs are public 05..12, one diagnostic-off and one diagnostic-on run
each, PP_PROFILE=1 and PP_GRAPH=0. The runner preserves AC/TLE results and requires
correct output even if diagnostic overhead exceeds the official limit. WA, RE,
missing counters, count invariant failures and wall timeouts stop the script.
Optional arguments select fewer inputs, e.g. `bash tests/run_evolution.sh cases/11.txt`.
Each measured process has a 30-second wall timeout. Five minutes is an allocation
ceiling, not a target runtime. Old benchmark scripts force PP_EVOLVE=0.

Logs under tmp/evolution-JOB-TIMESTAMP-PID:

- session.log: gate results, baseline/diagnostic phases, compact summary.
- evolution.tsv: one row per case/snapshot, counts, fractions, timing intervals.
- summary.tsv: baseline/diagnostic Compute, scan cost, final fractions, observed fixed point.
- baseline/ and evolution/: original results.tsv, program stderr and checker logs.

Paste session.log's final summary and evolution.tsv; no rerun is needed to inspect
them. A job interrupted by disconnect can be checked with `squeue -u "$USER"`;
existing log directories can be listed with `ls -dt tmp/evolution-*`.

## Graph-enabled official judge (separate from diagnostics)

```bash
make hw3-1 &&
env -u PP_KERNEL -u PP_ACTIVE -u PP_R1 -u PP_MOORE \
  PP_GRAPH=1 PP_EVOLVE=0 PP_PROFILE=0 hw3-1-judge
```

Do not wrap judge in srun; judge allocates its own GPU. Per-command env does not
persistently export PP_GRAPH into the login shell. This requests Graphs through
the judge's environment; the judge table alone does not prove that its wrapper
forwarded the variable. Preserve any launch/environment information the judge
provides. The explicit srun graph experiment already validated graph selection.
Respect the course full-judge cooldown rather than repeating immediately.

## Decision rules after measurement

- Low late work fraction while scheduling remains dense: investigate dynamic
  dense-to-sparse switching, including scan/queue-build and stale-buffer costs.
- An observed exact fixed point well before T: investigate exact on-device
  convergence detection; snapshots alone are not an implemented early-exit policy.
- Stable high work/changed fractions: keep updating, investigate representation,
  temporal reuse or kernel instruction/memory efficiency.
- Sparse work remains low but interval time high: investigate queue/atomic/block
  scheduling overhead and granularity. This report does not measure atomic stalls.

These diagnostics do NOT establish HBM bandwidth saturation, shared-memory bank
conflicts, achieved occupancy, or competitors' algorithms. Those require separate
targeted hardware measurements; no performance improvement is promised here.
