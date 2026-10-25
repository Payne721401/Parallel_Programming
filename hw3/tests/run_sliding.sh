#!/usr/bin/env bash
# Explicit SAT/sliding comparison, independent of default policy. Slurm only.
set -euo pipefail
export PP_GRAPH=0 # Preserve the non-graph baseline for this existing experiment.
cd "$(dirname "$0")/.."
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run this script through srun/sbatch on Nano5.' >&2
    exit 2
fi
mode=${1:-compare}
if [[ $mode != compare && $mode != ab && $mode != baseline ]]; then
    echo 'usage: bash tests/run_sliding.sh [compare|ab|baseline]' >&2
    exit 2
fi
command -v hw3-1-check >/dev/null
[[ -x ./hw3-1 ]]
logdir="tmp/sliding-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
git status --short
nvidia-smi --query-gpu=name,driver_version --format=csv

run_public() {
    local label=$1 method=$2 profile=$3 status=0
    echo "Public 05/09: PP_MOORE=$method PP_PROFILE=$profile, automatic scheduling"
    env -u PP_KERNEL -u PP_ACTIVE -u PP_R1 PP_MOORE="$method" PP_PROFILE="$profile" \
        python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 15 \
        --outdir "$logdir/$label" cases/05.txt cases/09.txt || status=$?
    python3 tests/check_timing_results.py "$logdir/$label/results.tsv" "$status"
}

if [[ $mode == baseline ]]; then
    # Optional diagnosis before trying the candidate: one measured SAT run only.
    run_public baseline sat 1
else
    [[ -x ./tests/test_hw3-1 ]]
    echo 'Gate 1: sliding dense vs CPU reference, 48 radius/boundary/state cases'
    env -u PP_R1 PP_KERNEL=tiled PP_ACTIVE=0 PP_MOORE=sliding PP_PROFILE=0 \
        timeout 40s ./tests/test_hw3-1 --sliding
    echo 'Gate 2: sliding sparse, cleanup/growth/fallback and von Neumann regression, 30 cases'
    env -u PP_R1 PP_KERNEL=tiled PP_ACTIVE=1 PP_MOORE=sliding PP_PROFILE=0 \
        timeout 40s ./tests/test_hw3-1 --sparse
    # Same-allocation ABBA. Both methods use the same instrumentation setting.
    profile=1
    if [[ $mode == ab ]]; then profile=0; fi
    for round in 1 2 3 4; do
        method=sat
        if [[ $round == 2 || $round == 3 ]]; then method=sliding; fi
        run_public "$round-$method" "$method" "$profile"
    done
    python3 tests/summarize_sliding.py "$logdir" "$profile"
fi
echo "Outputs verified correct; inspect AC/TLE and timings in $logdir (no default changed)"
