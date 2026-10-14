# HW2-2 Hot-Row Experiment

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
