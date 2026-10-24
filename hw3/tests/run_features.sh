#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run this script through srun/sbatch on Nano5.' >&2
    exit 2
fi
mode=${1:-profile}
if [[ $mode != profile && $mode != trace ]]; then
    echo 'usage: bash tests/run_features.sh profile | trace MATRIX_LOGDIR' >&2
    exit 2
fi
command -v hw3-1-check >/dev/null
unset PP_KERNEL PP_ACTIVE PP_R1 PP_PROFILE PP_MOORE
if [[ $mode == profile ]]; then
    [[ -x ./hw3-1 && -x ./tests/test_hw3-1 ]]
    logdir="tmp/features-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
else
    [[ $# == 2 && -d $2/inputs && -f $2/summary.tsv && -x ./tests/trace_hw3-1 ]]
    matrix_dir=$2
    command -v nsys >/dev/null
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
