#!/usr/bin/env bash
# Run inside one short allocation, after compiling on the login node.
set -euo pipefail
export PP_SUPPORT=tile # Preserve the reference influence policy.
export PP_EVOLVE=0 # Evolution scans must not contaminate performance comparisons.
cd "$(dirname "$0")/.."
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run this script through srun/sbatch on Nano5.' >&2
    exit 2
fi
logdir="tmp/baseline-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
git status --short
nvidia-smi --query-gpu=name,driver_version --format=csv
if [[ -f cases/01.txt ]]; then
    python3 tests/case_stats.py cases/[0-9][0-9].txt | tee "$logdir/case-stats.tsv"
fi
timeout 80s ./tests/test_hw3-1 64 20261022
python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 5 --outdir "$logdir/examples"
