#!/usr/bin/env bash
# Diagnostic only: samples full state but never changes evolution or stops early.
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -n ${SLURM_JOB_ID:-} ]] || fail 'Run through srun/sbatch on Nano5.'
command -v hw3-1-check >/dev/null || fail 'Missing hw3-1-check in PATH'
[[ -x ./hw3-1 && -x ./tests/test_evolution ]] || fail 'Run make hw3-1 tests/test_evolution first'
if [[ $# == 0 ]]; then set -- cases/{05,06,07,08,09,10,11,12}.txt; fi
for input in "$@"; do
    [[ -r $input ]] || fail "Missing input: $input"
    [[ -r ${input%.txt}.golden.txt || -r ${input%.txt}.out ]] || fail "Missing golden: $input"
done
unset PP_KERNEL PP_ACTIVE PP_R1 PP_MOORE PP_EVOLVE_EVERY
export PP_GRAPH=0 PP_PROFILE=0 PP_EVOLVE=0
logdir="tmp/evolution-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
git status --short
nvidia-smi --query-gpu=name,driver_version --format=csv
echo 'Gate 1: dense sampled counters vs CPU, 20 cases'
PP_ACTIVE=0 PP_EVOLVE=1 PP_EVOLVE_EVERY=128 timeout 60s ./tests/test_evolution > "$logdir/gate-dense.log" 2>&1 || { cat "$logdir/gate-dense.log"; exit 1; }
python3 tests/evolution_report.py --gate "$logdir/gate-dense.log"
echo 'Gate 2: sparse every-generation counters vs CPU, 20 cases'
PP_ACTIVE=1 PP_EVOLVE=1 PP_EVOLVE_EVERY=1 timeout 60s ./tests/test_evolution > "$logdir/gate-sparse.log" 2>&1 || { cat "$logdir/gate-sparse.log"; exit 1; }
python3 tests/evolution_report.py --gate "$logdir/gate-sparse.log"
for mode in baseline evolution; do
    enabled=0
    if [[ $mode == evolution ]]; then enabled=1; fi
    echo "$mode: PP_EVOLVE=$enabled PP_PROFILE=1 PP_GRAPH=0"
    status=0
    PP_EVOLVE=$enabled PP_PROFILE=1 python3 tests/run_cases.py --checker hw3-1-check \
        --wall-timeout 30 --outdir "$logdir/$mode" "$@" || status=$?
    python3 tests/check_timing_results.py "$logdir/$mode/results.tsv" "$status"
done
python3 tests/evolution_report.py "$logdir"
echo "Outputs verified; diagnostic runs can exceed timing limits. Inspect $logdir/evolution.tsv and summary.tsv."
