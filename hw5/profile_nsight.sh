#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mode=${1:-systems}
case_id=${2:-09}
phase=${3:-decode}
kernel=${4:-small_linear_kernel}
device=${5:-0}
if [[ ! $case_id =~ ^(0[1-9]|1[0-2])$ || ! $phase =~ ^(prefill|decode)$ ||
      ! $device =~ ^[01]$ || ! $kernel =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
    echo "Usage: bash profile_nsight.sh systems|compute CASE [prefill|decode KERNEL GPU]" >&2
    exit 2
fi
case "$mode" in
    systems) tool=nsys ;;
    compute) tool=ncu ;;
    *) echo "Choose systems or compute" >&2; exit 2 ;;
esac

export PATH="/work/HPC_software/LMOD/nvidia/packages/cuda-12.4/bin:$PATH"
command -v "$tool"
command -v hw5-check
make hw5_nsight
report=$(mktemp -d "$PWD/nsight-${mode}-${case_id}-XXXXXX")
echo "Report directory: $report"
{
    git rev-parse HEAD
    "$tool" --version
    echo "$mode case=$case_id phase=$phase kernel=$kernel device=$device"
} > "$report/config.txt"
app=(./hw5_nsight weights.bin "cases/$case_id.txt" "$report/out.bin")
allocation=(srun -A ACD115218 -p dev -N1 -n1 -c4 --gpus-per-node=2 -t 00:06:00)
if [[ -n ${HW5_EXCLUDE_NODE:-} ]]; then
    allocation+=(--exclude="$HW5_EXCLUDE_NODE")
fi
if [[ $mode == systems ]]; then
    profiler=(nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --output="$report/timeline")
else
    profiler=(ncu --nvtx --nvtx-include "$phase/" --devices "$device" \
        --kernel-name-base function --kernel-name "regex:(^|::)${kernel}$" \
        --launch-count 1 --set full --replay-mode kernel \
        --export "$report/kernel")
fi
"${allocation[@]}" bash -ec '
    echo "job=$SLURM_JOB_ID node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS-unset}"
    python3 - <<PY
import ctypes
import sys
try:
    rt = ctypes.CDLL("libcudart.so.12")
except OSError:
    rt = ctypes.CDLL("/work/HPC_software/LMOD/nvidia/packages/cuda-12.4/lib64/libcudart.so.12")
count = ctypes.c_int()
status = rt.cudaGetDeviceCount(ctypes.byref(count))
print(f"Before Nsight: cudaGetDeviceCount status={status}, visible={count.value}", flush=True)
if status or count.value < 2:
    sys.exit("Two CUDA devices required. Stopping before Nsight and weight loading.")
PY
    exec "$@"
' bash "${profiler[@]}" "${app[@]}" 2>&1 | tee "$report/run.log"
hw5-check "goldens/$case_id.golden.bin" "$report/out.bin" | tee "$report/check.log"
if [[ $mode == systems ]]; then
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
        "$report/timeline.nsys-rep" | tee "$report/summary.txt"
    python3 trace_experts.py "$report/timeline.sqlite" | tee "$report/overlap.txt"
else
    # A name/phase without a matching launch can exit without a usable report.
    test -s "$report/kernel.ncu-rep"
    ncu --import "$report/kernel.ncu-rep" --page details > "$report/summary.txt"
fi
echo "Report directory: $report"
