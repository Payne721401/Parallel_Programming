# HW2-2 Submission Gate

Submission defaults are PP_HOT=1, PP_HOT_MB=256, PP_ACCUM=auto and
PP_HOT_DENSE=0. These reproduce the configuration with a reported 17/17 AC,
1.61-second judge run; they do not promise that runtime or hidden-case success.
Explicit environment overrides remain available. Dense and new scheduling
experiments are not promoted for submission.

## Cluster Checks

Commit and push, then:

```bash
cd ~/pp_course && git pull --ff-only && cd hw2/hw2-2
jid=$(sbatch --parsable -o hw2-2-release-%j.log ../tests/batch_hw2-2.sh \
  'bash ../tests/run_hw2-2-release.sh')
echo "Job: $jid"
tail -F "hw2-2-release-$jid.log"
```

The wrapper clears all implementation PP_* overrides and forces a normal
rebuild. It stops at the first failed stage:

- In-memory multiply checks with UBSan and standard-library assertions on two
  nodes, one rank each. Checks submission defaults and small boundary shapes.
- Actual public c04 through the fixed driver on two nodes/two ranks, three times.
- All 14 public correctness cases at 1, 3, 4, 2, 5 and 8 ranks (84 checks).
- All 11 custom cases at 1, 3 and 8 ranks (33 checks), including >2^32 products.

Expected: 120 driver/checker runs, plus 18 in-memory checks. These are practice
allocation shapes, not 24-rank performance validation. In-memory checks isolate
multiply from MPI file output; they do not reproduce the actual public c04 input.
Passing them does not explain an old unreproduced segfault or prove no memory bug.

View results (Ctrl+C only stops tail):

```bash
cat "hw2-2-release-$jid.log"
d="tmp/hw2-2-release-$jid"
cat "$d/boundary.log"
# If any stage failed, inspect its stderr and checker diagnostics:
find "$d" -type f \( -name '*.err' -o -name '*.check' \) -size +0c \
  -print -exec tail -n 30 {} \;
```

Require all stages to pass and the final batch step status to be `ok`, not just
`== done`. If jid was lost: `log=$(ls -t hw2-2-release-*.log | head -n1)`, then
`jid=${log#hw2-2-release-}; jid=${jid%.log}`.

## Final Judge

Only after the gate passes, judge the exact defaults, without dense experiments:

```bash
module load gcc/15.2.0 openmpi/5.0.10
set -o pipefail
env -u PP_HOT -u PP_HOT_MB -u PP_HOT_REPORT -u PP_ACCUM -u PP_HOT_DENSE \
  hw2-2-judge 2>&1 | tee ~/judge-hw2-2-release.log
cat ~/judge-hw2-2-release.log
```

The practice QoS does not provide a 24-rank allocation. Public judge results
cover that shape, not every custom boundary there. Do not override the practice
batch shape to request unavailable resources or mistake local oversubscription
for a two-node judge test. Submit only hw2-2.cpp through the course workflow.
