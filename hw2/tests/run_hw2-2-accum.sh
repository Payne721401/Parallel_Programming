#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../hw2-2"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "submit through ../tests/batch_hw2-2.sh; do not run on the login node" >&2
    exit 1
fi
root="${OUT_ROOT:-tmp/hw2-2-accum-$SLURM_JOB_ID}"
ref="${AB_REF:-ff051af}"
cases="${CASES:-p01 p02 p03}"
correctness="c01 c02 c03 c04 c05 c06 c07 c08 c09 c10 c11 c12 c13 c14"
mkdir -p "$root/bin"
export PP_HOT=1 PP_HOT_MB="${PP_HOT_MB:-256}" PP_HOT_REPORT=0 TIMING=0
echo "== accumulator experiment: baseline=$ref; normal-build A/B; results=$root =="
mpicxx -std=c++17 -O3 -march=native "$script_dir/test_hw2-2-accum.cpp" -o "$root/bin/test-accum"
(ulimit -d 1228800; srun -N1 -n1 -c1 --hint=nomultithread "$root/bin/test-accum" </dev/null)

for mode in scan auto; do
    export PP_ACCUM="$mode"
    echo "== accumulator=$mode: correctness =="
    CUSTOM_CASES="" OUT_ROOT="$root/check-$mode" bash "$script_dir/run_hw2-2.sh" custom
    CASES="$correctness" OUT_ROOT="$root/check-$mode" bash "$script_dir/run_hw2-2.sh" public
    echo "== accumulator=$mode: normal-build A/B =="
    AB_REF="$ref" REPEAT="${REPEAT:-3}" CASES="$cases" OUT_ROOT="$root/ab-$mode" \
        bash "$script_dir/run_hw2-2.sh" ab
done

# Profile the current bitmap control too; profile times are not the A/B score.
for mode in bitmap scan auto; do
    echo "== accumulator=$mode: diagnostic profile =="
    PP_ACCUM="$mode" PP_HOT_REPORT=1 TIMING=1 CASES="$cases" OUT_ROOT="$root/profile-$mode" \
        bash "$script_dir/run_hw2-2.sh" public
done
echo "== accumulator experiment complete: $root =="
