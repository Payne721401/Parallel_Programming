#!/usr/bin/env bash
# One candidate depth per allocation; avoid a blind depth sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z ${SLURM_JOB_ID:-} ]]; then echo 'Run through srun/sbatch.' >&2; exit 2; fi
mode=${1:-compare}
depth=${2:-2}
if [[ $mode != compare && $mode != ab && $mode != check ]] || [[ ! $depth =~ ^(2|4|8)$ ]]; then
    echo 'usage: bash tests/run_temporal.sh [compare|ab|check] [2|4|8]' >&2
    exit 2
fi
command -v hw3-1-check >/dev/null
[[ -x ./hw3-1 && -x ./tests/test_hw3-1 ]]
unset PP_KERNEL PP_ACTIVE PP_R1 PP_MOORE PP_SUPPORT PP_EVOLVE_EVERY
export PP_GRAPH=0 PP_EVOLVE=0 PP_PROFILE=0 PP_TEMPORAL="$depth"
logdir="tmp/temporal-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
git status --short
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset} depth=$depth"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo 'Gate 1: dense temporal vs CPU, 168 cases; includes short-run fallbacks'
PP_ACTIVE=0 PP_PROFILE=1 timeout 60s ./tests/test_hw3-1 --temporal > "$logdir/gate.log" 2>&1 || {
    cat "$logdir/gate.log"; exit 1;
}
python3 tests/summarize_temporal.py gate "$logdir/gate.log" "$depth"
echo 'Gate 2: temporal request with sparse scheduling uses one-step bbox, 30 cases'
PP_ACTIVE=1 timeout 40s ./tests/test_hw3-1 --sparse
echo 'Gate 3: direct/tiled fallback and default bbox, 22 cases'
PP_KERNEL=tiled PP_ACTIVE=0 timeout 40s ./tests/test_hw3-1 --quick
if [[ $mode == check ]]; then exit 0; fi

profile=1
if [[ $mode == ab ]]; then profile=0; fi
for round in 1 2 3 4; do
    k=1
    if [[ $round == 2 || $round == 3 ]]; then k=$depth; fi
    echo "Round $round: PP_TEMPORAL=$k PP_PROFILE=$profile"
    status=0
    PP_TEMPORAL="$k" PP_PROFILE="$profile" \
        python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 15 \
        --outdir "$logdir/$round-$k" cases/11.txt cases/12.txt || status=$?
    python3 tests/check_timing_results.py "$logdir/$round-$k/results.tsv" "$status"
done
python3 tests/summarize_temporal.py compare "$logdir" "$depth" "$profile"
echo "Logs: $logdir; temporal remains opt-in; bbox is the normal default."
