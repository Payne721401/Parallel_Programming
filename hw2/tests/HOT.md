# HW2-2 Hot-Row Experiment

For the subsequent bitmap/scan accumulator A/B, see [ACCUM.md](ACCUM.md).

Only `hw2-2.cpp` changes the submitted program. A and B remain immutable CSR;
C retains A's original row ownership and canonical CSR ordering.

## Candidate

- `PP_HOT=0` (default): original K-split path.
- `PP_HOT=1`: count references to each B row, multiply by its length, and select
  high-work rows from overloaded owners. Cache entries are bounded by
  `PP_HOT_MB` (default 256 MiB per node, range 0..512; zero disables the cache).
- One MPI shared window per node holds the selected B rows, ordered by global k.
  Node leaders exchange the packed ranges. Node membership comes from MPI,
  not an assumed contiguous rank layout.
- Assign each output row's hot work as one task, largest first, starting from
  the ranks' cold-work totals. Cold A entries still go to their original B owner.
- A single rank, empty inputs, fewer than 2^20 total products, initial imbalance
  at most 1.25, or a predicted maximum-work reduction under roughly 10% use
  the old path. These are initial heuristics, not measured optimal thresholds.
- Each A entry goes to exactly one worker. Cached B rows are exact copies, and
  all partial results still use the original exact accumulator and merge.

`PP_HOT_REPORT=1` prints `[hw2-2-hot]` in a normal build; it is enabled by default
in the profiling build. `enabled=1` is only permission to plan: `selected_rows`
must be positive to confirm the cache was actually used. `planned_max` is a
product-count prediction, not a runtime prediction. Skipped discovery can report
zero planning counts; profiling's `products` is the actual assigned work.

## Cluster Run

Commit and push the changes first. On SEIREN:

```bash
cd ~/pp_course
git pull --ff-only && git merge-base --is-ancestor 3ed8a35 HEAD || exit 1
git log -1 --oneline
git status --short
cd hw2/hw2-2
jid=$(sbatch --parsable -o hw2-2-hot-%j.log ../tests/batch_hw2-2.sh \
  'export PP_HOT=1 PP_HOT_MB=256 PP_HOT_REPORT=1
   export OUT_ROOT=tmp/hw2-2-hot-$SLURM_JOB_ID
   bash ../tests/run_hw2-2.sh custom &&
   CASES="c01 c02 c03 c04 c05 c06 c07 c08 c09 c10 c11 c12 c13 c14" bash ../tests/run_hw2-2.sh public &&
   AB_REF=1604d20 CASES="p01 p02 p03" REPEAT=3 bash ../tests/run_hw2-2.sh ab &&
   TIMING=1 CASES="p01 p02 p03" bash ../tests/run_hw2-2.sh public')
echo "Job: $jid; log: hw2-2-hot-$jid.log"
squeue -u "$USER"
tail -F "hw2-2-hot-$jid.log"
```

`tail -F` waits for a queued job's log to appear. Ctrl+C stops viewing, not the
job. After `== done`, require the step status `ok`. On a fresh output directory,
the four stages should report 33, 14, 18 and 3 passes, all with zero failures.
The new `hot` custom case exercises reuse concentrated away from rank zero;
custom also checks empty/uneven shapes and the existing dense and nnz boundaries.
Stages stop at the first failed script. A/B uses normal builds, alternates their
order, checks every output, and compares the candidate with commit `1604d20`.

Each invocation uses a fresh output directory, so consecutive steps do not
delete and recreate the same file across filesystem clients. Passing outputs
are removed; failed output paths are kept and printed in the run's `.err` log.
`NO` means srun exited successfully but the batch shell could not see the output;
it is not a checker mismatch (`WA`). If custom tests fail, A/B and profile files
will not exist because the remaining stages were not run.

View and share the complete log, raw A/B times and candidate phase reports:

```bash
cat "hw2-2-hot-$jid.log"
cat "tmp/hw2-2-hot-$jid/ab.tsv"
cat "tmp/hw2-2-hot-$jid"/logs/public_p0*.err
```

If the shell lost `$jid`, recover it from the newest matching log:

```bash
log=$(ls -t hw2-2-hot-*.log | head -n 1)
jid=${log#hw2-2-hot-}
jid=${jid%.log}
cat "$log"
```

After correctness passes, run the full judge and keep its log:

```bash
set -o pipefail
PP_HOT=1 PP_HOT_MB=256 PP_HOT_REPORT=0 hw2-2-judge 2>&1 | tee ~/judge-hw2-2-hot.log
cat ~/judge-hw2-2-hot.log
```

The judge must preserve the environment switch to exercise the candidate.
Default invocation still runs the old path. The A/B reports directly confirm
activation, but do not establish that a separate judge run inherited its flag.

## Read the Measurements

- `hot-plan`: reference/length discovery, selection and weighted assignment.
- `hot-local`: node discovery, allocation and copying locally owned hot rows.
- `hot-share`: leader broadcasts and publication to node-local readers.
- `products max/mean`: actual remaining work imbalance; compare with `planned_max`.
- `c-bytes`, `c-xnode`: whether partial-result traffic grew after redistributing.
- `hot-release`: cache teardown (and preceding unmarked cleanup).

Phase MPI_MAX values may come from different ranks; never sum them into elapsed
time. Exchange includes waiting. Eight-rank diagnostics are not 24-rank judge
times, and shared-page RSS is not private-memory usage. Only normal-build A/B
and judge runs establish end-to-end performance.

Next changes depend on these results: reduce planning/copy cost if preparation
dominates, refine assignment if work remains skewed, investigate partial-result
traffic if exchange grows, and tune the accumulator only after compute is both
dominant and reasonably balanced. Do not change several of these at once.

## Per-Round Diagnostics

`TIMING=1` also records every rank of every round. No new barrier or collective
is inserted into the round loop; records are gathered at the end and only rank
zero prints them. The normal build contains none of this instrumentation.

- `[hw2-2-round-summary]`: round number, global row interval `[lo, hi)`, total
  products, maximum/mean products and its rank, minimum/maximum compute time
  and the slowest compute rank. The work-heavy and slowest ranks may differ.
- `[hw2-2-round]`: one record per rank per round. `node` identifies the lowest
  world rank on that node. All `_s` fields are local elapsed seconds, never
  cross-rank timestamp differences. All `send_*` and `recv_*` fields are bytes.
- `a_entries`, `hot_products`, `cold_products`: assigned A entries (including
  those referencing empty B rows) and exact scalar-product counts. These are
  recorded from row bounds, not counted inside the scalar multiplication loop.
- `partial_rows`, `partial_nnz`: nonempty partial rows and their stored entries,
  before the final merge. Sent bytes equal `8 * (partial_rows + partial_nnz)`;
  each partial row has an additional 8-byte header.
- `compute_s` includes the round's bound scan, output preparation, accumulation
  and partial-row emission. `prepare_s` covers output clear/reserve and send-count
  reset; `accumulate_s` covers each row's A traversal, B lookup and accumulator
  updates; `emit_s` covers its header insertion and sorted extraction/reset.
  These subfields are nested within compute, not extra elapsed time. Loop and
  timer overhead are not fully attributed to the subfields.
- `counts_s`: receive-count initialization and MPI_Alltoall of counts;
  `layout_s`: allocation and construction of MPI counts/displacements;
  `alloc_s`: allocation/zero-initialization of the receive buffer;
  `data_s`: MPI_Alltoallv of partial C; `merge_s`: merging into final C.
- Traffic is split into `self`, `local` (other ranks on the same node), and
  `xnode` (another node), for both sending and receiving. Their sums match the
  older C traffic counters. Rank totals reveal concentrated receivers too.

MPI call durations can include waiting; `data_s` is NOT pure network time.
Whole-run phase maxima and per-round compute scopes differ (setup/cleanup may
fall outside a round). Do not add phase maxima or nested subfields together.
Profiling totals include row timers, extra bound accounting and final reporting;
only normal builds and judge establish performance. No additional speedup is
expected from this diagnostics-only change. `PP_HOT` still defaults to zero.

After committing and pushing, run just p01-p03 twice at the practice shape:

```bash
cd ~/pp_course
git pull --ff-only
cd hw2/hw2-2
jid=$(sbatch --parsable -o hw2-2-round-%j.log ../tests/batch_hw2-2.sh \
  'export PP_HOT=1 PP_HOT_MB=256 PP_HOT_REPORT=1 TIMING=1
   export CASES="p01 p02 p03"
   for rep in 1 2; do
     echo "== round diagnostics repetition $rep =="
     OUT_ROOT=tmp/hw2-2-round-$SLURM_JOB_ID/r$rep \
       bash ../tests/run_hw2-2.sh public || exit 1
   done')
echo "Job: $jid; log: hw2-2-round-$jid.log"
tail -F "hw2-2-round-$jid.log"
```

After `== done`, expect two `3 passed, 0 failed` summaries and final status `ok`.
Ctrl+C stops viewing, not the job. View and share all results:

```bash
cat "hw2-2-round-$jid.log"
d="tmp/hw2-2-round-$jid"
for f in "$d"/r*/logs/public_p0*.err; do
  echo "== $f =="
  cat "$f"
done
```

If `$jid` was lost, recover it before those commands:

```bash
log=$(ls -t hw2-2-round-*.log | head -n 1)
jid=${log#hw2-2-round-}
jid=${jid%.log}
cat "$log"
```
