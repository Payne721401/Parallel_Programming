#!/usr/bin/env bash
# One allocation: correctness, profiled comparison, then unprofiled comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z ${SLURM_JOB_ID:-} ]]; then echo 'Run through srun/sbatch on Nano5.' >&2; exit 2; fi
mode=${1:-compare}
if [[ $mode != compare && $mode != check ]]; then echo 'usage: bash tests/run_bitplane.sh [compare|check]' >&2; exit 2; fi
command -v hw3-1-check >/dev/null
[[ -x ./hw3-1 && -x ./tests/test_hw3-1 ]]
unset PP_KERNEL PP_ACTIVE PP_R1 PP_MOORE PP_SUPPORT PP_EVOLVE_EVERY
export PP_GRAPH=0 PP_EVOLVE=0 PP_PROFILE=1 PP_BITPLANE=1 PP_BIT_ROWS=4 PP_BIT_LOGIC=reference
logdir="tmp/bitplane-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
git status --short
sha256sum hw3-1 tests/test_hw3-1 hw3-1.cu r1_bits.hpp r1_bitplane.cuh
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
nvidia-smi --query-gpu=name,driver_version --format=csv
if [[ -f tmp/bitplane-resources.log ]]; then cp tmp/bitplane-resources.log "$logdir/compiler-resources.log"; fi

for rows in 1 4; do
    echo "Dense CPU gate: 452 cases, rows=$rows"
    PP_ACTIVE=0 PP_BIT_ROWS="$rows" timeout 90s ./tests/test_hw3-1 --bitplane > "$logdir/gate-$rows.log" 2>&1 || {
        cat "$logdir/gate-$rows.log"; exit 1;
    }
    python3 tests/summarize_bitplane.py gate "$logdir/gate-$rows.log" "$rows" dense
done
for kind in sparse graphs tiled; do
    echo "Fallback CPU gate: $kind"
    args=(env PP_ACTIVE=1)
    flag="--$kind"
    if [[ $kind == graphs ]]; then args+=(PP_GRAPH=1); fi
    if [[ $kind == tiled ]]; then args=(env PP_ACTIVE=0 PP_KERNEL=tiled); flag=--quick; fi
    "${args[@]}" timeout 90s ./tests/test_hw3-1 "$flag" > "$logdir/gate-$kind.log" 2>&1 || {
        cat "$logdir/gate-$kind.log"; exit 1;
    }
    python3 tests/summarize_bitplane.py gate "$logdir/gate-$kind.log" 4 "$kind"
done
if [[ $mode == check ]]; then echo "Correctness complete: $logdir"; exit 0; fi

for profile in 1 0; do
    round=0
    for method in byte bits1 bits4 bits4 bits1 byte; do
        round=$((round+1)); bits=1; rows=4
        if [[ $method == byte ]]; then bits=0; fi
        if [[ $method == bits1 ]]; then rows=1; fi
        echo "Profile=$profile round=$round method=$method (automatic scheduling)"
        status=0
        PP_BITPLANE="$bits" PP_BIT_ROWS="$rows" PP_PROFILE="$profile" \
            python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 20 \
            --outdir "$logdir/profile-$profile/$round-$method" \
            cases/01.txt cases/11.txt cases/12.txt || status=$?
        python3 tests/check_timing_results.py "$logdir/profile-$profile/$round-$method/results.tsv" "$status"
    done
    python3 tests/summarize_bitplane.py compare "$logdir/profile-$profile" "$profile"
done
echo "Logs: $logdir; inspect profile-0/summary.tsv, profile-1/summary.tsv and compiler-resources.log."
echo 'Bit-plane remains opt-in. Full H2D/D2H and conversion are included in Compute time.'
