#!/usr/bin/env bash
# One bounded allocation: correctness first, then only unfinished public 08-10.
set -euo pipefail
export PP_SUPPORT=tile # Preserve the reference influence policy.
export PP_EVOLVE=0 # Evolution scans must not contaminate performance comparisons.
export PP_GRAPH=0 # Preserve the non-graph baseline for this existing experiment.
cd "$(dirname "$0")/.."
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run this script through srun/sbatch on Nano5.' >&2
    exit 2
fi
mode=${1:-profile}
unset PP_MOORE # Use current defaults, not an inherited manual override.
if [[ $mode != check && $mode != profile && $mode != gate ]]; then
    echo 'usage: bash tests/run_sparse.sh [profile|check|gate]' >&2
    exit 2
fi
[[ -x ./hw3-1 && -x ./tests/test_hw3-1 ]]
if [[ $mode != gate ]]; then command -v hw3-1-check >/dev/null; fi
logdir="tmp/sparse-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
git status --short
nvidia-smi --query-gpu=name,driver_version --format=csv

echo 'Gate 1: dense refactor, 22 CPU-reference comparisons'
env -u PP_KERNEL -u PP_R1 PP_ACTIVE=0 PP_PROFILE=0 timeout 15s ./tests/test_hw3-1 --quick
echo 'Gate 2: forced sparse entry, 22 rule/radius/boundary comparisons'
env -u PP_KERNEL -u PP_R1 PP_ACTIVE=1 PP_PROFILE=0 timeout 15s ./tests/test_hw3-1 --quick
echo 'Gate 3: sparse motion, decay, stale-buffer cleanup, and growth fallback, 30 comparisons'
env -u PP_KERNEL -u PP_R1 PP_ACTIVE=1 PP_PROFILE=0 timeout 25s ./tests/test_hw3-1 --sparse
if [[ $mode == gate ]]; then
    echo "Correctness gates passed; logs: $logdir"
    exit 0
fi

profile=0
if [[ $mode == profile ]]; then profile=1; fi
echo "Public cases: PP_KERNEL/PP_ACTIVE unset (judge defaults), PP_PROFILE=$profile"
for case_id in 08 09 10; do
    status=0
    env -u PP_KERNEL -u PP_ACTIVE -u PP_R1 PP_PROFILE=$profile python3 tests/run_cases.py \
        --checker hw3-1-check --wall-timeout 15 --outdir "$logdir/$case_id" \
        "cases/$case_id.txt" || status=$?
    # Correct-but-slow outputs are useful diagnostics. Never proceed after
    # WA, RE, missing output, timeout, or a failed checker/tool invocation.
    python3 tests/check_timing_results.py "$logdir/$case_id/results.tsv" "$status"
done
echo "All public outputs verified correct; inspect AC/TLE in $logdir/{08,09,10}/results.tsv"
