# HW2-2 Accumulator Experiment

Only `hw2-2.cpp` changes the submitted program. Scheduling, hot-cache selection,
communication and final-C merge remain unchanged. The experiment changes only
the accumulation and extraction of partial C rows.

## Modes

- `PP_ACCUM=bitmap`: existing word-batched bitmap accumulator.
- `PP_ACCUM=scan`: direct uint32 multiply-add without bitmap maintenance; scan
  all N columns in order, append nonzeros, and clear the accumulator after each
  partial row. This remains exact even for sparse or empty output.
- `PP_ACCUM=auto` (submission default): scan when that rank's partial row has at least `8*N` scalar
  products, otherwise bitmap. This initial heuristic is NOT a measured optimum
  and does not assume that the output is dense. It uses runtime work and shape,
  never case names. Selection is made once per output row; the scalar loop has
  no mode branch. Both paths reset all their touched state before switching.

The final merge always uses bitmap. `PP_HOT` now defaults to one; the wrapper
enables it equally for baseline and candidates. `PP_HOT_REPORT=1` also prints
`[hw2-2-accum]` with the selected mode. Profiling adds `acc-scan-rows`,
`acc-bitmap-rows` and `acc-scan-products`; row counters exclude zero-product rows.
Their row sum equals the number of emitted partial rows.

## Run on SEIREN

Commit and push first. Then:

```bash
cd ~/pp_course && git pull --ff-only && cd hw2/hw2-2
jid=$(sbatch --parsable -o hw2-2-accum-%j.log ../tests/batch_hw2-2.sh \
  'bash ../tests/run_hw2-2-accum.sh')
echo "Job: $jid; log: hw2-2-accum-$jid.log"
tail -F "hw2-2-accum-$jid.log"
```

The default workflow is:

1. A small direct accumulator test: bitmap/scan switching, reset, empty, dense
   and sparse rows, 64-bit bitmap boundaries, auto threshold and large sums.
2. For each of scan and auto: 33 custom checks against the starter, 14 public
   correctness checks, then 18 alternating normal-build A/B checks against
   `ff051af` with `PP_HOT=1` (three repetitions of p01-p03). That saved baseline
   does not implement `PP_ACCUM`, so the environment cannot turn it into scan.
3. One p01-p03 profile per bitmap/scan/auto mode (three passes per mode).

Expect 139 checked MPI runs with zero failures, plus the direct test. Every
failed stage stops the experiment. Require final status `ok` in the batch log;
`== done` alone is insufficient. Ctrl+C stops log viewing, not the job.
`AB_REF`, `REPEAT`, `CASES` and `OUT_ROOT` can override the wrapper defaults; only
use a baseline revision known to have the bitmap path when changing `AB_REF`.

## View Results

After completion, share the batch log, raw A/B tables and compact phase reports:

```bash
cat "hw2-2-accum-$jid.log"
d="tmp/hw2-2-accum-$jid"
for mode in scan auto; do
  echo "== $mode A/B =="
  cat "$d/ab-$mode/ab.tsv"
done
for f in "$d"/profile-*/logs/public_p0*.err; do
  echo "== $f =="
  grep -E '^\[hw2-2(-accum)?\]' "$f"
done
```

Full per-round diagnostics remain in those `.err` files. If `$jid` was lost:

```bash
log=$(ls -t hw2-2-accum-*.log | head -n 1)
jid=${log#hw2-2-accum-}
jid=${jid%.log}
cat "$log"
```

Only normal-build A/B and judge establish speedup. Compare accumulate/emit times
and scan coverage in profiles, but do not add nested phase maxima or treat MPI
durations as pure network time. This is an 8-rank experiment, not a prediction
of the 24-rank judge. Check correctness and A/B before choosing a judge mode;
no default is promoted by this experiment.
