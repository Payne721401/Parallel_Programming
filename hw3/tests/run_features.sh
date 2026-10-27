#!/usr/bin/env bash
set -euo pipefail
export PP_BITPLANE=0 # Preserve the established experiment baseline.
export PP_SUPPORT=tile # Preserve the reference influence policy.
export PP_EVOLVE=0 # Evolution scans must not contaminate performance comparisons.
export PP_GRAPH=0 # Preserve the non-graph baseline for this existing experiment.
cd "$(dirname "$0")/.."
fail() { echo "ERROR: $*" >&2; exit 1; }
require_command() {
    command -v "$1" >/dev/null || fail "Cannot find '$1' in PATH; check the modules/environment in this login session."
}
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run this script through srun/sbatch on Nano5.' >&2
    exit 2
fi
mode=${1:-profile}
if [[ $mode != profile && $mode != trace ]]; then
    echo 'usage: bash tests/run_features.sh profile | trace MATRIX_LOGDIR' >&2
    exit 2
fi
require_command hw3-1-check
unset PP_KERNEL PP_ACTIVE PP_R1 PP_PROFILE PP_MOORE
if [[ $mode == profile ]]; then
    [[ -x ./hw3-1 ]] || fail 'Missing executable ./hw3-1; run make hw3-1.'
    [[ -x ./tests/test_hw3-1 ]] || fail 'Missing executable ./tests/test_hw3-1; run make tests/test_hw3-1.'
    logdir="tmp/features-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
else
    [[ $# == 2 ]] || fail 'usage: bash tests/run_features.sh trace MATRIX_LOGDIR'
    matrix_dir=$2
    [[ -d "$matrix_dir/inputs" ]] || fail "Missing matrix input directory: $matrix_dir/inputs"
    [[ -f "$matrix_dir/summary.tsv" ]] || fail "Missing completed matrix summary: $matrix_dir/summary.tsv"
    [[ -x ./tests/trace_hw3-1 ]] || fail 'Missing executable ./tests/trace_hw3-1; run make tests/trace_hw3-1.'
    require_command nsys
    for name in r1_large r1_cluster; do
        for suffix in txt golden.txt; do
            [[ -r "$matrix_dir/inputs/$name.$suffix" ]] || fail "Missing or unreadable matrix file: $matrix_dir/inputs/$name.$suffix"
        done
    done
    logdir="tmp/features-trace-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
fi
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
git status --short
nvidia-smi --query-gpu=name,driver_version --format=csv
if [[ $mode == profile ]]; then
    echo 'Default dispatch: 22 quick CPU-reference checks'
    PP_PROFILE=0 timeout 30s ./tests/test_hw3-1 --quick
    echo 'Forced sparse default dispatch: 30 CPU-reference regression checks'
    PP_ACTIVE=1 PP_PROFILE=0 timeout 40s ./tests/test_hw3-1 --sparse
    python3 tests/feature_matrix.py --outdir "$logdir"
    echo "Matrix complete: $logdir/summary.tsv and $logdir/samples.tsv"
else
    nsys --version
    for name in r1_large r1_cluster; do
        active=auto
        if [[ $name == r1_cluster ]]; then active=1; fi
        echo "Diagnostic trace: $name PP_ACTIVE=$active; do not use this timing as a benchmark"
        env PP_ACTIVE=$active PP_PROFILE=0 timeout 60s nsys profile \
            --trace=cuda --sample=none --cpuctxsw=none \
            --capture-range=cudaProfilerApi --capture-range-end=stop \
            --output="$logdir/$name" \
            ./tests/trace_hw3-1 "$matrix_dir/inputs/$name.txt" "$logdir/$name.out"
        timeout 10s hw3-1-check "$matrix_dir/inputs/$name.golden.txt" "$logdir/$name.out"
        timeout 40s nsys stats \
            --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
            "$logdir/$name.nsys-rep" > "$logdir/$name.stats.txt"
        cat "$logdir/$name.stats.txt"
    done
    echo 'Inspect .nsys-rep timelines for launch gaps; API durations overlap GPU work and must not be added to kernel totals.'
fi
