#!/usr/bin/env bash
# Factorial comparison: comparator logic x spatial rows; defaults unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run through srun/sbatch.' >&2; exit 2; }
mode=${1:-compare}
[[ $mode == compare || $mode == check ]] || { echo 'usage: run_bitlogic.sh [compare|check]' >&2; exit 2; }
command -v hw3-1-check >/dev/null
[[ -x hw3-1 && -x tests/test_hw3-1 ]]
unset PP_KERNEL PP_ACTIVE PP_R1 PP_MOORE PP_SUPPORT PP_EVOLVE_EVERY
export PP_GRAPH=0 PP_EVOLVE=0 PP_PROFILE=1 PP_BITPLANE=1 PP_BIT_ROWS=4 PP_BIT_LOGIC=reference
logdir="tmp/bitlogic-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
git status --short
sha256sum hw3-1 tests/test_hw3-1 hw3-1.cu r1_bits.hpp r1_bitplane.cuh
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
nvidia-smi --query-gpu=name,driver_version --format=csv
if [[ -f tmp/bitplane-resources.log ]]; then cp tmp/bitplane-resources.log "$logdir/compiler-resources.log"; fi
for logic in reference compact; do
    for rows in 1 2 4; do
        path="$logdir/gate-$logic-$rows.log"
        echo "CPU reference gate: logic=$logic rows=$rows (452 cases)"
        PP_BIT_LOGIC="$logic" PP_BIT_ROWS="$rows" PP_ACTIVE=0 \
            timeout 90s ./tests/test_hw3-1 --bitplane > "$path" 2>&1 || { cat "$path"; exit 1; }
        python3 tests/summarize_bitlogic.py gate "$path" "$rows" "$logic" dense
    done
done
for kind in sparse graphs tiled; do
    args=(env PP_ACTIVE=1 PP_BIT_LOGIC=compact PP_BIT_ROWS=2)
    flag="--$kind"
    if [[ $kind == graphs ]]; then args+=(PP_GRAPH=1); fi
    if [[ $kind == tiled ]]; then args+=(PP_ACTIVE=0 PP_KERNEL=tiled); flag=--quick; fi
    path="$logdir/gate-$kind.log"
    "${args[@]}" timeout 90s ./tests/test_hw3-1 "$flag" > "$path" 2>&1 || { cat "$path"; exit 1; }
    python3 tests/summarize_bitlogic.py gate "$path" 2 compact "$kind"
done
if [[ $mode == check ]]; then echo "Correctness complete: $logdir"; exit 0; fi
methods=(reference1 reference2 reference4 compact1 compact2 compact4 compact4 compact2 compact1 reference4 reference2 reference1)
for profile in 1 0; do
    round=0
    for method in "${methods[@]}"; do
        round=$((round+1)); rows=${method: -1}; logic=${method%?}
        echo "Profile=$profile round=$round logic=$logic rows=$rows"
        status=0
        PP_BIT_LOGIC="$logic" PP_BIT_ROWS="$rows" PP_PROFILE="$profile" \
            python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 20 \
            --outdir "$logdir/profile-$profile/$round-$method" \
            cases/01.txt cases/11.txt cases/12.txt || status=$?
        python3 tests/check_timing_results.py "$logdir/profile-$profile/$round-$method/results.tsv" "$status"
    done
    python3 tests/summarize_bitlogic.py compare "$logdir/profile-$profile" "$profile"
done
echo "Logs: $logdir; profile-0 is end-to-end timing, profile-1 includes dispatch/phase evidence."
echo 'Two samples/configuration, retain outliers. No default changed; speedup is not guaranteed.'
