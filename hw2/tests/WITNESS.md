# HW2-1 Exact Witness Experiment

For the independent bucket/CAS diagnostics and sampled-witness experiment, see
[SEARCH.md](SEARCH.md). Both use isolated binaries and leave defaults unchanged.

Exact pruning is **on by default**, with tau=16, in `hw2-1.cpp`.
`PP_WITNESS=off` retains the original path. The measurement package does not change `main.cpp`, the
generator, the Makefile, HW2-2, or the existing test runner.

## Step 2: Pruning Light Edges

`PP_WITNESS_LIGHT=1` also tests edges with weight <= tau for redundancy. It is
disabled by default. The light graph and its exact distances are computed as
before. On any edge of a shortest-path tree rooted at the pivot,
`L[child] = L[parent] + w`, so `L[parent] + L[child] >= w`. Such edges cannot be
deleted by the strict test. A whole witness tree therefore survives even when
other light edges are removed. Unreachable vertices still use the saturated 255
sentinel and cannot trigger pruning. `PP_WITNESS=probe` only counts candidates.

After committing, pushing, and pulling in `~/pp_course`, run this targeted A/B
from `~/pp_course/hw2/hw2-1`. The baseline is Step 1 at `020d319`; it ignores the
new environment variable. Both variants print counts into their `.err` logs.

```bash
sbatch -o hw2-1-light-%j.log ../tests/batch_hw2-1.sh \
  'AB_REF=020d319 CASES="07 10 12 13" REPEAT=3 \
   PP_WITNESS=prune PP_WITNESS_TAU=16 PP_WITNESS_LIGHT=1 PP_WITNESS_REPORT=1 \
   OUT_ROOT=tmp/hw2-1-light-$SLURM_JOB_ID bash ../tests/run_hw2-1.sh ab'
```

Require `24 passed, 0 failed` before interpreting times. For the same input and
tau, the decrease in `kept_edges` from base to new is precisely the number of
newly removed light edges. The candidate report includes `light_pruning=1`.
For job JOBID, inspect `tmp/hw2-1-light-JOBID/ab/*.1.err` for first-run counts.

A full judge run of the experimental path can be requested explicitly:

```bash
PP_WITNESS=prune PP_WITNESS_TAU=16 PP_WITNESS_LIGHT=1 PP_WITNESS_REPORT=0 \
  hw2-1-judge 2>&1 | tee ~/judge-light.log
```

The environment switch must reach the judged process for that run to exercise
Step 2. The direct `srun` A/B above also records which mode actually ran.

## Run on SEIREN

Synchronize `hw2-1/hw2-1.cpp` and these four files in `hw2/tests`:
`run_hw2-1-witness.sh`, `batch_hw2-1-witness.sh`,
`summarize_hw2-1-witness.py`, and this document. Keep Unix line endings in scripts.
The cluster checkout needs revision `5c8b402`, the non-bitmap F2 baseline.

From the cluster's `hw2/hw2-1` directory:

```bash
sbatch ../tests/batch_hw2-1-witness.sh
squeue -u "$USER"
tail -f hw2-1-witness-JOBID.log
```

The default is one 8-rank, 32 GiB, 40-minute practice allocation. No judge command
is called. All executables run through `srun`. Each run has a three-minute limit
and the 1200 MiB private-data limit. The original submission executable is not
overwritten. Public inputs/goldens come from `/home/share/pp26/cases/hw2-1`.
Cases 09/10 are included; if the practice allocation cannot accommodate a case,
it is recorded as a failure, not silently omitted from the total.

Default sequence:

1. Compile F2 baseline, current normal build, phase-only profiling build, and the
   sequential starter reference into an isolated timestamped directory.
2. Compare six small boundary cases against the starter at 1, 3, and 8 ranks.
   Test five cutoffs, probe/prune, and the batch/relabel/generator switches.
   Stop public timing on any correctness failure.
3. Run **all 17 public cases**, three normal-build repetitions each for baseline,
   candidate off, and candidate prune. Rotate their order. Check every output.
4. For each public case, run phase-only off/probe/prune once, also checked.
5. Write `runs.tsv`, `summary.tsv`, `summary.md`, `phases.tsv`, `counts.tsv`, raw
   logs, source snapshots and metadata under `tmp/hw2-1-witness/job-*`.
   A sibling `.tar.gz` contains everything except executables and binary outputs.

For a quicker first pass over **all 17 cases**, use one repetition:

```bash
REPEAT=1 sbatch ../tests/batch_hw2-1-witness.sh
```

After the first pass, cutoff sensitivity can be measured over the same full set:

```bash
TAU=8 sbatch ../tests/batch_hw2-1-witness.sh
# After that job completes:
TAU=32 sbatch ../tests/batch_hw2-1-witness.sh
```

Other overrides: `PUBLIC_NP` (1..8), `CASES="01 04"`, `CASES_DIR` (the directory
containing `01.txt`, not its parent), `BASE_REF`, `OUT_ROOT`, `STEP_TIME`.
All inherited `PP_*` variables are cleared for controlled comparisons.
If the job times out, retain its directory/log; do not accept incomplete sums.

## What the Experiment Tests

Let H contain all original edges of weight <= tau. Choose the highest-degree
vertex in H and compute shortest-path lengths L from it. Keep all H edges.
Remove an original edge (u,v,w) only when both L values are finite and
`uint64_t(L[u]) + L[v] < w`. A strictly shorter u-to-landmark-to-v walk then exists
using only retained edges. Thus all-pairs distances and connectivity are
preserved, including disconnected graphs; infinite endpoints are never pruned.

The implementation generates twice: first store only H, compute L and release
H; then regenerate and retain only necessary edges before constructing the final
CSR. The timing includes both generations, preparation, filtering and releases.
`PP_WITNESS=probe` pays preparation/filter costs but keeps the full graph.
`PP_WITNESS=prune` actually prunes. `PP_WITNESS_TAU` defaults to 16 (range 1..255).

Use normal-build medians to judge speedup. Diagnostic times explain its source,
but must not be mixed with normal times. These 8-rank results do not establish
24-rank judge performance. The default was enabled after the reported 151 checked
cluster runs passed; a complete official judge run is still needed for the total score.
Other proposals (tiled CSR fill, new SIMD relaxation, narrower synchronization)
are not implemented or measured by this package.
