#!/usr/bin/env bash
# Same-allocation sparse influence ABBA. No GPU work outside Slurm; bbox is the normal default.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run through srun/sbatch on Nano5.' >&2
    exit 2
fi
mode=${1:-compare}
if [[ $mode != compare && $mode != ab ]]; then
    echo 'usage: bash tests/run_support.sh [compare|ab]' >&2
    exit 2
fi
command -v hw3-1-check >/dev/null
[[ -x ./hw3-1 && -x ./tests/test_hw3-1 ]]
for case in 08 09 10; do
    [[ -f cases/$case.txt && -f cases/$case.golden.txt ]]
done
unset PP_KERNEL PP_ACTIVE PP_R1 PP_MOORE PP_EVOLVE_EVERY
export PP_GRAPH=0 PP_EVOLVE=0 PP_PROFILE=0 PP_SUPPORT=bbox
logdir="tmp/support-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
git status --short
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo 'Gate 1: bbox default dispatch; support/decay/cleanup/R32, 28 CPU comparisons'
PP_ACTIVE=1 timeout 90s ./tests/test_hw3-1 --support
echo 'Gate 2: bbox packed R1; aligned/unaligned words and tails, 64 CPU comparisons'
PP_ACTIVE=1 PP_R1=packed timeout 40s ./tests/test_hw3-1 --r1
echo 'Gate 3: direct/SAT/sliding bbox paths; each 30 CPU sparse comparisons'
PP_ACTIVE=1 PP_KERNEL=naive timeout 40s ./tests/test_hw3-1 --sparse
PP_ACTIVE=1 PP_KERNEL=tiled PP_MOORE=sat timeout 40s ./tests/test_hw3-1 --sparse
PP_ACTIVE=1 PP_KERNEL=tiled PP_MOORE=sliding timeout 40s ./tests/test_hw3-1 --sparse
echo 'Gate 4: bbox option with dense execution; 22 CPU comparisons'
PP_ACTIVE=0 timeout 40s ./tests/test_hw3-1 --quick
echo 'Gate 5: bbox graph replay/parity/epoch regression; 38 CPU comparisons'
PP_ACTIVE=1 PP_GRAPH=1 timeout 40s ./tests/test_hw3-1 --graphs

profile=1
if [[ $mode == ab ]]; then profile=0; fi
for round in 1 2 3 4; do
    method=tile
    if [[ $round == 2 || $round == 3 ]]; then method=bbox; fi
    echo "Round $round: PP_SUPPORT=$method PP_PROFILE=$profile; initial schedule unchanged"
    status=0
    PP_SUPPORT="$method" PP_PROFILE="$profile" \
        python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 15 \
        --outdir "$logdir/$round-$method" cases/08.txt cases/09.txt cases/10.txt || status=$?
    python3 tests/check_timing_results.py "$logdir/$round-$method/results.tsv" "$status"
done
python3 tests/summarize_support.py "$logdir" "$profile"
echo "Logs: $logdir; bbox is the normal default. Correctness is separate from timing."
