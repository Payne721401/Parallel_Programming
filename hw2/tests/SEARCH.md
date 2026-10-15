# HW2-1 Bucket, Atomic And Sampled-Witness Experiments

No new optimization is enabled by default. `PP_WITNESS_SAMPLE=1` keeps the full
witness preparation; `PP_WITNESS_LIGHT` remains opt-in. Only `hw2-1.cpp` is a
submission file. The driver, generator, Makefile and HW2-2 are unchanged.

## Run On SEIREN

After committing/pushing the changes locally:

```bash
cd ~/pp_course && git pull && cd hw2/hw2-1
module load gcc/15.2.0 openmpi/5.0.10
sbatch -o hw2-1-search-%j.log ../tests/batch_hw2-1.sh \
  'bash ../tests/run_hw2-1-search.sh all'
```

Default: cases 07, 10, 12, 13, eight ranks on one practice node, three repetitions
for each normal-build sampling variant. All variants use prune, tau=16 and
light pruning enabled. Inherited `PP_*` values are cleared. A source snapshot,
independent executables, compiler/CPU metadata and raw results are saved in a
unique directory. The existing `hw2-1` executable/objects are not overwritten.

Alternatively submit the two experiments independently:

```bash
sbatch -o hw2-1-bucket-%j.log ../tests/batch_hw2-1.sh \
  'bash ../tests/run_hw2-1-search.sh bucket'
sbatch -o hw2-1-sample-%j.log ../tests/batch_hw2-1.sh \
  'bash ../tests/run_hw2-1-search.sh sample'
```

Use either the combined job OR these separate jobs; there is no need to run both.
Each normal A/B comparison stays within its own allocation. Do not compare raw
times across the two jobs. `CASES`, `REPEAT`, `PUBLIC_NP`, `CASES_DIR`, `OUT_ROOT`
and `STEP_TIME` can be overridden. `CASES` takes two-digit IDs, not `all`.
For example `CASES="09 10" bash ../tests/run_hw2-1-search.sh sample`.

Every executable runs through `srun`, with a five-minute step timeout and a
1200 MiB private-data cap. The allocation is the existing 8-rank, 32 GiB practice
configuration, not the 24-rank judge. Sampling may weaken pruning enough to
increase memory use or fail a large case; the script reports this as a failure.

## Results

Replace JOBID with the ID printed by sbatch; for the combined job:

```bash
tail -f hw2-1-search-JOBID.log
# After completion:
d=$(find tmp/hw2-1-search -maxdepth 1 -type d -name 'all-JOBID-*' -print -quit)
cat "$d/exit-status" "$d/summary.md"
cat "$d/runs.tsv"
cat "$d/witness.tsv"
cat "$d/phases.tsv"
cat "$d/counts.tsv"
cat "$d/invariants.tsv"
```

For separate jobs use the corresponding `hw2-1-bucket-JOBID.log` or
`hw2-1-sample-JOBID.log`, and `bucket-JOBID-*` or `sample-JOBID-*` directories.
Require `exit-status` equal to 0, the `passed, 0 failed` line, complete planned
runs and all invariant checks passing. The batch wrapper itself continues after
a failed step, so its final `== done` alone does not prove success. A timeout may
leave no final summary; keep the raw logs and do not accept an incomplete run.

Small cases are checked against the original sequential implementation before
public runs: full sampled distance vectors, repeated/isolated sources, P>N,
scramble on/off and maximum seed/edgefactor. Sampling also exercises the official
generator, no relabeling, batch=1, light pruning off and tau=255. Every public
output is checked against its official golden.

## What Changes

`PP_WITNESS_SAMPLE=2` or `4` constructs H only from original edge indices
`[0, ceil(M / divisor))`, retaining edges with weight <= tau. This deterministic
prefix is independent of rank count and keeps the existing contiguous/SIMD
generator path. It is not a claim of unbiased random sampling. The final graph
still generates all M original edges and applies the same exact pruning test.

Correctness does not require the sampled H to approximate the original graph.
Distances L are exact in H. Every shortest-path-tree edge of H survives the
strict test `L[u] + L[v] < w`, since on a tree edge the sum is at least w. The
surviving tree provides a strictly shorter replacement walk for each deleted
edge. Unreachable or saturated bounds cannot trigger deletion. A sparse/empty
sample can therefore reduce the pruning benefit, but must not change answers.

The sample experiment rotates normal-build full/half/quarter witness runs,
then profiles each divisor once. These medians include all preparation and
cleanup. Inspect `witness_generated`, `light_reached`, `kept_edges`, total time
and memory together: cheaper preparation alone is not a successful optimization.

## Diagnostic Builds

- `phase`: `HW21_PROF` + `HW21_PHASE_ONLY`; existing phase/work counts, no new
  per-push or per-CAS accounting.
- `bucket`: also `HW21_BUCKET_PROF`; visits, stale removals, published entries,
  vector growths, bytes copied on growth and capacities released. `bucket:grow`
  times only pushes that expand a vector, including allocation/copy/free and
  clock overhead. It is nested within search, mostly within `trav:relax`.
- `atomic`: also `HW21_ATOMIC_PROF`; CAS attempts, failures, changed expected
  values, abandoned updates after failure and relaxations needing no CAS.

Bucket and atomic diagnostics use the full witness (divisor=1). No pool or new
queue algorithm is implemented yet. Normal builds compile these counters out.
The diagnostic builds perturb execution order and cache behavior; their times
and counter differences are not an optimization A/B result.

`counts.tsv` separates main search from light-witness search. Useful identities:

```text
pushes + bucket-seeds = bucket-stale + bucket-published
repeated valid inspections = bucket-inspected - bucket-stale - bucket-published
cas-attempts = pushes + cas-failed
arcs = cas-skipped + pushes + cas-abandoned
```

The summarizer checks these identities. `bucket-stale` is obsolete queued work,
not repeated adjacency expansion. `cas-failed / cas-attempts` measures failed
attempts, not time lost. Changed expected values prove concurrent decreases;
unchanged failures may be spurious because the algorithm uses weak CAS. A low
failure rate does not rule out cache-line ownership transfers, false sharing,
NUMA latency or expensive successful atomics. No CAS instruction is removed.

## Delta And Partitioning

This experiment does not change Delta. Current search is exact integer-distance
Dial-style search (Delta=1), with all ranks sharing each frontier's arc work.
Using a larger Delta needs bucket closure or another correct ordering scheme;
changing the ring size or the witness tau is not sufficient. Heavy edges are
deferred until bucket closure, not automatically pruned by Delta-stepping.

The current architecture works on rank 0's shared-memory node. It does not send
one MPI message per relaxed edge. Owner hashing and degree-based relabeling are
already used, but do not prove NUMA-optimal placement. METIS is neither a required
ingredient of Delta-stepping nor an automatic win; partitioning, graph migration
and possible load imbalance must be included in any comparison.

After the sampling results justify a candidate, run a full judge explicitly,
for example for divisor 2 (do not assume 2 is the winner):

```bash
PP_WITNESS=prune PP_WITNESS_TAU=16 PP_WITNESS_LIGHT=1 \
  PP_WITNESS_SAMPLE=2 PP_WITNESS_REPORT=1 hw2-1-judge 2>&1 | tee ~/judge-sample2.log
```

The direct-srun diagnostics record the actual divisor. The judge switch must
also reach the executable. Keep default=1 until correctness and normal-build
end-to-end measurements justify a change.
