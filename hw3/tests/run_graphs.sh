#!/usr/bin/env bash
# First experiment: graph submission only, retaining all existing update kernels.
set -euo pipefail
export PP_EVOLVE=0 # Evolution scans must not contaminate performance comparisons.
cd "$(dirname "$0")/.."
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -n ${SLURM_JOB_ID:-} ]] || fail 'Run through srun/sbatch on Nano5.'
[[ $# -ge 1 && $# -le 2 ]] || fail 'usage: bash tests/run_graphs.sh MATRIX_LOGDIR [compare|ab]'
matrix=$1
mode=${2:-compare}
[[ $mode == compare || $mode == ab ]] || fail 'mode must be compare (profiled) or ab (unprofiled)'
[[ -f "$matrix/manifest.json" && -f "$matrix/summary.tsv" ]] || fail "Missing completed matrix: $matrix"
for name in r1_short r1_long r1_c4 r1_large r1_cluster moore_r5; do
    for suffix in txt golden.txt; do
        [[ -r "$matrix/inputs/$name.$suffix" ]] || fail "Missing input/golden: $name.$suffix"
    done
done
command -v hw3-1-check >/dev/null || fail 'hw3-1-check is missing from PATH'
[[ -x ./hw3-1 && -x ./tests/test_hw3-1 ]] || fail 'Run make hw3-1 tests/test_hw3-1 first'
unset PP_KERNEL PP_ACTIVE PP_R1 PP_MOORE PP_GRAPH PP_PROFILE
logdir="tmp/graphs-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
echo "Matrix inputs: $matrix"
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
git status --short
nvidia-smi --query-gpu=name,driver_version --format=csv
echo 'Gate 1: graph off, forced sparse; 38 CPU-reference checks'
PP_GRAPH=0 PP_ACTIVE=1 PP_PROFILE=0 timeout 60s ./tests/test_hw3-1 --graphs
echo 'Gate 2: graph on, dense; 38 CPU-reference checks'
PP_GRAPH=1 PP_ACTIVE=0 PP_PROFILE=0 timeout 60s ./tests/test_hw3-1 --graphs
echo 'Gate 3: graph on, sparse; 38 CPU-reference checks'
PP_GRAPH=1 PP_ACTIVE=1 PP_PROFILE=0 timeout 60s ./tests/test_hw3-1 --graphs
echo 'Gate 4: graph on, SAT sparse fallback path; 38 CPU-reference checks'
PP_GRAPH=1 PP_KERNEL=tiled PP_MOORE=sat PP_ACTIVE=1 PP_PROFILE=0 timeout 60s ./tests/test_hw3-1 --graphs
profile=1
if [[ $mode == ab ]]; then profile=0; fi
python3 tests/compare_graphs.py "$matrix" "$logdir" --profile "$profile"
echo "Inspect $logdir/session.log and summary.tsv; GPU validation is complete only if all gates/comparisons pass."
