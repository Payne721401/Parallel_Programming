#!/usr/bin/env bash
# Diagnostic tooling only: no production source/default changes.
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "ERROR: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null || fail "Missing $1; load cuda/12.4 in this login session."; }
if [[ -z ${SLURM_JOB_ID:-} ]]; then fail 'Run through srun/sbatch, not on the login node.'; fi
mode=${1:-}; [[ $# -eq 0 ]] || shift
case "$mode" in
    cases|nsys|ncu) ;;
    *) fail 'usage: bash tests/inspect_bitplane.sh cases|nsys|ncu [CASE ...], default 11' ;;
esac
if [[ $# -eq 0 ]]; then set -- 11; fi
for case_id in "$@"; do
    [[ $case_id =~ ^(0[1-9]|1[0-2])$ ]] || fail "Use a two-digit public case ID: $case_id"
    [[ -f cases/$case_id.txt && -f cases/$case_id.golden.txt ]] || fail "Missing input/golden for $case_id"
done
require hw3-1-check
[[ -x hw3-1 ]] || fail 'Run make hw3-1 first.'
if [[ $mode == nsys ]]; then
    require nsys
    [[ -x tests/trace_hw3-1 ]] || fail 'Run make tests/trace_hw3-1 first.'
elif [[ $mode == ncu ]]; then require ncu; fi
unset PP_KERNEL PP_ACTIVE PP_R1 PP_MOORE PP_SUPPORT PP_EVOLVE_EVERY
export PP_GRAPH=0 PP_EVOLVE=0 PP_PROFILE=0
export PP_BITPLANE=${PP_BITPLANE:-auto} PP_BIT_ROWS=${PP_BIT_ROWS:-4}
export PP_BIT_LOGIC=${PP_BIT_LOGIC:-compact}
[[ $PP_BITPLANE == auto || $PP_BITPLANE == 0 || $PP_BITPLANE == 1 ]] || fail 'Invalid PP_BITPLANE'
[[ $PP_BIT_ROWS == 1 || $PP_BIT_ROWS == 2 || $PP_BIT_ROWS == 4 ]] || fail 'Invalid PP_BIT_ROWS'
[[ $PP_BIT_LOGIC == reference || $PP_BIT_LOGIC == compact ]] || fail 'Invalid PP_BIT_LOGIC'
logdir="tmp/inspect-bitplane-${mode}-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
git status --short
sha256sum hw3-1 hw3-1.cu r1_bits.hpp r1_bitplane.cuh
echo "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-unset}"
nvidia-smi --query-gpu=name,driver_version --format=csv

if [[ $mode == cases ]]; then
    inputs=(); for case_id in "$@"; do inputs+=("cases/$case_id.txt"); done
    echo 'One correctness/dispatch observation per case, automatic schedule. NOT an A/B comparison.'
    status=0
    PP_PROFILE=1 python3 tests/run_cases.py --checker hw3-1-check --wall-timeout 20 \
        --outdir "$logdir/cases" "${inputs[@]}" || status=$?
    python3 tests/check_timing_results.py "$logdir/cases/results.tsv" "$status"
    python3 - "$logdir/cases" <<'PY'
import csv,os,sys
from pathlib import Path
sys.path.insert(0,'tests')
from summarize_bitplane import records,verify_public_dispatch
root=Path(sys.argv[1])
with (root/'results.tsv').open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
for i,row in enumerate(rows):
    case=Path(row['case']).stem
    data=records((root/f'{i:02d}-{case}.log').read_text(),'PP_BITPLANE')
    if len(data)!=1: raise ValueError('Missing/duplicate bit-plane diagnostic; rebuild binary')
    r=data[0]
    header=Path(row['case']).read_text().splitlines()
    verify_public_dispatch(r,row,header,os.environ['PP_BITPLANE'],int(os.environ['PP_BIT_ROWS']),os.environ['PP_BIT_LOGIC'])
    print(f"{case}: {row['verdict']} kernel={row['profile_kernel']} bitplane_selected={r['selected']} reason={r['reason']}")
PY
    echo "Logs: $logdir; TLE with correct output is retained, not called a performance pass."
    exit 0
fi

"$mode" --version
echo "PP_BITPLANE=$PP_BITPLANE PP_BIT_ROWS=$PP_BIT_ROWS PP_BIT_LOGIC=$PP_BIT_LOGIC"
if [[ $mode == nsys ]]; then sha256sum tests/trace_hw3-1; fi
methods=(selected)
if [[ $mode == nsys ]]; then methods=(byte bits1 bits4); fi
# ncu profiles the selected configuration ONCE per case, including fallbacks.
kernel_filter='step_(r1_(dense|bits)|sparse|tiled|naive)'
for case_id in "$@"; do
    for method in "${methods[@]}"; do
        if [[ $mode == nsys ]]; then
            export PP_BITPLANE=1 PP_BIT_ROWS=4
            if [[ $method == byte ]]; then export PP_BITPLANE=0; fi
            if [[ $method == bits1 ]]; then export PP_BIT_ROWS=1; fi
        fi
        prefix="$logdir/$case_id-$method"
        echo "Diagnostic $mode: case=$case_id method=$method; timings are NOT judge measurements."
        if [[ $mode == nsys ]]; then
            timeout 90s nsys profile --trace=cuda --sample=none --cpuctxsw=none \
                --capture-range=cudaProfilerApi --capture-range-end=stop \
                --output="$prefix" ./tests/trace_hw3-1 "cases/$case_id.txt" "$prefix.out" \
                > "$prefix.capture.log" 2>&1 || { cat "$prefix.capture.log"; fail 'nsys capture failed'; }
        else
            # Skip 20 MATCHING evolution kernels; pack/init kernels do not count.
            # Profile three launches and let the remaining simulation finish for checking.
            timeout 150s ncu --kernel-name-base demangled \
                --kernel-name "regex:$kernel_filter" --launch-skip 20 --launch-count 3 \
                --section SpeedOfLight --section LaunchStats --section Occupancy \
                --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
                --export "$prefix" ./hw3-1 "cases/$case_id.txt" "$prefix.out" \
                > "$prefix.capture.log" 2>&1 || {
                    cat "$prefix.capture.log"
                    fail 'ncu failed. ERR_NVGPUCTRPERM requires administrator-enabled counters; do not use sudo or bypass policy.'
                }
        fi
        timeout 10s hw3-1-check "cases/$case_id.golden.txt" "$prefix.out" > "$prefix.check.log" 2>&1 || {
            cat "$prefix.check.log"; fail 'Profiled output failed correctness check';
        }
        if [[ $mode == nsys ]]; then
            timeout 60s nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
                "$prefix.nsys-rep" > "$prefix.stats.txt" 2>&1 || { cat "$prefix.stats.txt"; fail 'nsys stats failed'; }
        else
            [[ -s "$prefix.ncu-rep" ]] || fail 'No ncu report; check kernel filter and capture log.'
            timeout 60s ncu --import "$prefix.ncu-rep" --page details > "$prefix.stats.txt" 2>&1 || {
                cat "$prefix.stats.txt"; fail 'ncu report import failed';
            }
            grep -Eq "$kernel_filter" "$prefix.stats.txt" || fail 'No matching kernel found in ncu report.'
        fi
        cat "$prefix.stats.txt"
        echo "Verified output: case=$case_id method=$method"
    done
done
echo "Logs: $logdir; .stats.txt for text, .nsys-rep/.ncu-rep for GUI. Do not sum API and GPU durations."
