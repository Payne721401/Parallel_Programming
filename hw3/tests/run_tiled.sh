#!/usr/bin/env bash
# Small correctness gate + selected public cases. Use 1 minute for check,
# or 2 minutes for the optional alternating A/B run after correctness passes.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run this script through srun/sbatch on Nano5.' >&2
    exit 2
fi
mode=${1:-check}
if [[ $mode != check && $mode != profile && $mode != ab ]]; then
    echo 'usage: bash tests/run_tiled.sh [check|profile|ab]' >&2
    exit 2
fi
command -v hw3-1-check >/dev/null
[[ -x ./hw3-1 && -x ./tests/test_hw3-1 ]]
logdir="tmp/tiled-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
profile=0
if [[ $mode == profile ]]; then profile=1; fi
echo "PP_PROFILE=$profile (public cases only; quick correctness checks use 0)"
if [[ $mode != ab ]]; then
    echo 'PP_KERNEL=unset: testing the judge default (auto: R>=5 tiled, otherwise naive)'
fi
git status --short
nvidia-smi --query-gpu=name,driver_version --format=csv
echo 'Correctness gate: default dispatch vs the original CPU reference, 22 small cases'
env -u PP_KERNEL -u PP_ACTIVE PP_PROFILE=0 timeout 20s ./tests/test_hw3-1 --quick

if [[ $mode != ab ]]; then
    # A first candidate check, not a causal A/B speedup measurement.
    env -u PP_KERNEL -u PP_ACTIVE PP_PROFILE=$profile python3 tests/run_cases.py --checker hw3-1-check \
        --wall-timeout 8 --outdir "$logdir/tiled" cases/05.txt cases/06.txt cases/07.txt
else
    # Only after the candidate passes. Limit the A/B run to 05 and 06:
    # naive 07 already hit TLE+ and is an unnecessary quota expense here.
    for round in 1 2 3 4; do
        kernel=naive
        if [[ $round == 2 || $round == 3 ]]; then kernel=tiled; fi
        echo "Round $round: PP_KERNEL=$kernel"
        # The baseline is known to TLE. Continue on timing failures, but stop
        # immediately on a wrong answer, runtime error, or wall timeout.
        status=0
        PP_KERNEL=$kernel PP_ACTIVE=0 PP_PROFILE=0 python3 tests/run_cases.py --checker hw3-1-check \
            --wall-timeout 8 --outdir "$logdir/$round-$kernel" cases/05.txt cases/06.txt || status=$?
        python3 tests/check_timing_results.py "$logdir/$round-$kernel/results.tsv" "$status"
    done
fi
