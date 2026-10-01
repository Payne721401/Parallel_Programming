#!/usr/bin/env bash
set -euo pipefail

# Run from the hw1 directory on SEIREN:
#   tests/run_public_smoke.sh baseline
#   tests/run_public_smoke.sh compare
#
# baseline: build and save outputs from the current code.
# compare:  build, rerun the same public smoke cases, run official checkers,
#           and compare against the saved baseline where bit-exactness is safe.

mode="${1:-smoke}"
case "$mode" in
    smoke|baseline|compare) ;;
    *)
        echo "usage: $0 [smoke|baseline|compare]" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw1_dir="$(cd "$script_dir/.." && pwd)"
cd "$hw1_dir"

threads="${THREADS:-8}"
cases_dir="${CASES_DIR:-cases}"
out_root="${OUT_ROOT:-tmp/hw1-smoke}"
baseline_dir="${BASELINE_DIR:-$out_root/baseline}"
run_dir="$out_root/latest"

hw11_case="${HW11_CASE:-t01}"
hw12_case="${HW12_CASE:-a01}"
hw13_case="${HW13_CASE:-p06}"

mkdir -p "$run_dir"
if [[ "$mode" == "baseline" ]]; then
    rm -rf "$baseline_dir"
    mkdir -p "$baseline_dir"
fi

require_path() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "missing required path: $path" >&2
        exit 1
    fi
}

if [[ "${ALLOW_NO_SLURM:-0}" == "1" ]]; then
    launcher=()
elif command -v srun >/dev/null 2>&1; then
    launcher=(srun -n 1 -c "$threads")
else
    echo "srun not found; set ALLOW_NO_SLURM=1 only for local syntax/debug runs" >&2
    exit 1
fi

run_cmd() {
    if [[ "${TIMING:-1}" == "1" ]]; then
        PP_TIMING=1 "${launcher[@]}" "$@"
    else
        "${launcher[@]}" "$@"
    fi
}

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "missing checker command: $cmd" >&2
        exit 1
    fi
}

build() {
    make clean
    make
}

official_checks() {
    check_command hw1-1-check
    check_command hw1-2-check
    check_command hw1-3-check

    hw1-1-check "$cases_dir/hw1-1/$hw11_case.out.png" "$run_dir/hw1-1-$hw11_case.out.png"
    hw1-2-check "$cases_dir/hw1-2/$hw12_case.out.txt" "$run_dir/hw1-2-$hw12_case.out.txt"
    hw1-3-check "$cases_dir/hw1-3/$hw13_case.out" "$run_dir/hw1-3-$hw13_case.out"
}

hw13_args() {
    local case_file="$cases_dir/hw1-3/$hw13_case.txt"
    require_path "$case_file"
    local args
    args="$( \
        (grep -Eo '[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?' "$case_file" || true) \
        | head -n 4 \
        | paste -sd ' ' - \
    )"
    if [[ "$(wc -w <<<"$args")" -ne 4 ]]; then
        echo "cannot parse N T seed theta from $case_file" >&2
        exit 1
    fi
    printf '%s\n' "$args"
}

run_cases() {
    require_path "$cases_dir/hw1-1/$hw11_case.in.png"
    require_path "$cases_dir/hw1-2/$hw12_case.in_A.png"
    require_path "$cases_dir/hw1-2/$hw12_case.in_B.png"

    echo "== hw1-1 $hw11_case =="
    run_cmd ./hw1-1 \
        "$cases_dir/hw1-1/$hw11_case.in.png" \
        "$run_dir/hw1-1-$hw11_case.out.png"

    echo "== hw1-2 $hw12_case =="
    run_cmd ./hw1-2 \
        "$cases_dir/hw1-2/$hw12_case.in_A.png" \
        "$cases_dir/hw1-2/$hw12_case.in_B.png" \
        "$run_dir/hw1-2-$hw12_case.out.txt"

    echo "== hw1-3 $hw13_case =="
    read -r n t seed theta <<<"$(hw13_args)"
    run_cmd ./hw1-3 "$n" "$t" "$seed" "$theta" "$run_dir/hw1-3-$hw13_case.out"
}

save_baseline() {
    cp "$run_dir/hw1-1-$hw11_case.out.png" "$baseline_dir/"
    cp "$run_dir/hw1-2-$hw12_case.out.txt" "$baseline_dir/"
    cp "$run_dir/hw1-3-$hw13_case.out" "$baseline_dir/"
    cat >"$baseline_dir/manifest.txt" <<EOF
THREADS=$threads
HW11_CASE=$hw11_case
HW12_CASE=$hw12_case
HW13_CASE=$hw13_case
EOF
}

compare_baseline() {
    require_path "$baseline_dir/manifest.txt"

    echo "== exact compare: hw1-1 PNG bytes, stricter than checker =="
    cmp "$baseline_dir/hw1-1-$hw11_case.out.png" "$run_dir/hw1-1-$hw11_case.out.png"

    echo "== exact compare: hw1-2 text output =="
    cmp "$baseline_dir/hw1-2-$hw12_case.out.txt" "$run_dir/hw1-2-$hw12_case.out.txt"

    echo "== exact compare: hw1-3 sample temperatures only =="
    diff -u \
        <(tail -n +3 "$baseline_dir/hw1-3-$hw13_case.out") \
        <(tail -n +3 "$run_dir/hw1-3-$hw13_case.out")
}

build
run_cases
official_checks

if [[ "$mode" == "baseline" ]]; then
    save_baseline
    echo "baseline saved to $baseline_dir"
elif [[ "$mode" == "compare" ]]; then
    compare_baseline
    echo "compare passed against $baseline_dir"
else
    echo "smoke passed"
fi
