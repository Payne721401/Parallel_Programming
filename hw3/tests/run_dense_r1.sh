#!/usr/bin/env bash
# Dense R1: default-dispatch correctness, sparse regression, then public 11/12 only.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run this script through srun/sbatch on Nano5.' >&2
    exit 2
fi
mode=${1:-profile}
unset PP_MOORE # Keep sliding experiments isolated in run_sliding.sh.
if [[ $mode != profile && $mode != check && $mode != ab ]]; then
    echo 'usage: bash tests/run_dense_r1.sh [profile|check|ab]' >&2
    exit 2
fi
command -v hw3-1-check >/dev/null
[[ -x ./hw3-1 && -x ./tests/test_hw3-1 ]]
logdir="tmp/dense-r1-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
git status --short
nvidia-smi --query-gpu=name,driver_version --format=csv
echo 'Gate 1: default dense R1 vs CPU reference, 90 boundary/distribution/long-run cases'
env -u PP_KERNEL -u PP_ACTIVE -u PP_R1 PP_PROFILE=0 timeout 25s ./tests/test_hw3-1 --dense-r1
echo 'Gate 2: sparse packed regression, 30 motion/decay/growth cases (includes R5)'
env -u PP_KERNEL PP_ACTIVE=1 PP_R1=packed PP_PROFILE=0 timeout 25s ./tests/test_hw3-1 --sparse

if [[ $mode == ab ]]; then
    # Optional controlled comparison: old direct / packed / packed / old direct.
    # Keep automatic scheduling identical; only change the R1 counting algorithm.
    for round in 1 2 3 4; do
        r1=direct
        if [[ $round == 2 || $round == 3 ]]; then r1=packed; fi
        echo "Round $round: PP_R1=$r1, profiling off"
        status=0
        env -u PP_KERNEL -u PP_ACTIVE PP_R1=$r1 PP_PROFILE=0 \
            python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 15 \
            --outdir "$logdir/$round-$r1" cases/11.txt cases/12.txt || status=$?
        python3 tests/check_timing_results.py "$logdir/$round-$r1/results.tsv" "$status"
    done
else
    profile=0
    if [[ $mode == profile ]]; then profile=1; fi
    echo "Public 11/12: judge defaults, PP_PROFILE=$profile (expect dense_r1 in profile mode)"
    status=0
    env -u PP_KERNEL -u PP_ACTIVE -u PP_R1 PP_PROFILE=$profile \
        python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 15 \
        --outdir "$logdir/default" cases/11.txt cases/12.txt || status=$?
    python3 tests/check_timing_results.py "$logdir/default/results.tsv" "$status"
fi
echo "Outputs verified correct; inspect AC/TLE and phase times in $logdir"
