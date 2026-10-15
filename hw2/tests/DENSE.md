# HW2-2 Dense Hot-Row Experiment

This tests `auto + dense hot rows`, not dense versus auto. Defaults are unchanged.
Only `hw2-2.cpp` changes the submitted program.

## Isolated Change

`PP_HOT_DENSE=1` builds an additional uint8 node-shared representation of selected
hot B rows with `4*nnz >= N`. It is used ONLY when the existing accumulator policy
selects scan for the partial C row. Bitmap rows, sparse hot rows, cold rows and
the final merge retain their existing paths. Values widen to uint32 before
multiplication and accumulation. No floating-point arithmetic is introduced.

Hot selection, worker assignment, CSR cache broadcasts and output communication
are unchanged. The shadow is built AFTER the original CSR share, so this first
experiment does not claim reduced network traffic. It deliberately retains the
CSR cache for fallback. Additional shared storage is at most half the existing
CSR cache cap (128 MiB with PP_HOT_MB=256); offsets cost 4*K bytes per rank when
at least one dense row is selected. Packing, allocation, synchronization and
release are included in end-to-end times. No cache is built in bitmap mode.

The 1/4 density cutoff is an initial bounded-work experiment: dense iterations
are at most four times the replaced nonzero products. It is not a tuned optimum.

## Run and Inspect

Commit and push first, then on the cluster:

```bash
cd ~/pp_course && git pull --ff-only && cd hw2/hw2-2
jid=$(sbatch --parsable -o hw2-2-dense-%j.log ../tests/batch_hw2-2.sh \
  'bash ../tests/run_hw2-2-dense.sh')
echo "Job: $jid; log: hw2-2-dense-$jid.log"
tail -F "hw2-2-dense-$jid.log"
```

Ctrl+C stops viewing, not the job. Expect a unit test, 33 custom checks, 14 public
correctness checks, 18 normal A/B checks, and 6 profile checks: 71 MPI checks.
Require zero failures AND final batch status `ok`; `== done` alone is insufficient.
Baseline d5b33a4 uses auto and ignores PP_HOT_DENSE. A/B alternates baseline and
candidate within one allocation; do not override AB_REF with a revision that
also enables dense through this same inherited environment.

After completion:

```bash
cat "hw2-2-dense-$jid.log"
d="tmp/hw2-2-dense-$jid"
cat "$d/ab/ab.tsv"
for f in "$d"/profile-*/logs/public_p0*.err; do
  echo "== $f =="
  grep -E '^\[hw2-2(-hot|-accum|-dense|-round-summary)?\]' "$f"
done
cat "$d/vectorization.txt"
```

If jid was lost: `log=$(ls -t hw2-2-dense-*.log | head -n1)`, then
`jid=${log#hw2-2-dense-}; jid=${jid%.log}`.

## Decision Metrics

- Normal A/B medians and raw repetitions decide end-to-end benefit. Profiles do
  not establish speedup. This practice allocation is 8 ranks, not the judge's 24.
- `hot-dense`: elapsed shadow allocation, conversion and publication, including
  arrival skew. `hot-release` also includes shadow teardown.
- `[hw2-2-dense]`: enabled flag, selected row count, shared bytes per node and
  offset bytes per rank. No candidate kernel ran if dense-products is zero.
- `dense-products / products`: fraction of original nonzero products replaced.
  `dense-slots / dense-products`: actual dense iterations per replaced product.
  `dense-calls` counts eligible A entries on scan-selected partial rows.
- Per-round `accumulate_s` includes both sparse and dense work (NOT an isolated
  dense-kernel timer). Compare it with `emit_s`, output counts and work balance.
  Per-phase maxima may refer to different ranks; never sum nested measurements.
- Product counts and partial-C traffic should match the baseline. The existing
  checker compares the full canonical output on every MPI run.
- Check the compiler report at the dense loop source line printed by the wrapper;
  unrelated vectorized loops do not prove this kernel was vectorized.

If the candidate passes, test formal performance separately:

```bash
PP_HOT=1 PP_HOT_MB=256 PP_ACCUM=auto PP_HOT_DENSE=1 PP_HOT_REPORT=0 \
  hw2-2-judge 2>&1 | tee ~/judge-hw2-2-dense.log
cat ~/judge-hw2-2-dense.log
```

This experiment does not change the default mode or claim a guaranteed speedup.
