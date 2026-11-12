#!/bin/bash
# One allocation, no full episodes. Each tool has a wall-time cap.
# Scaling is skipped by default; SCALING=1 repeats it. --check-tools needs no GPU.
#SBATCH -J hw4-gpu-diag
#SBATCH -A ACD115218
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-node=1
#SBATCH -t 00:05:00
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
INPUT=${INPUT:-$HOME/hw4-prof/b-452742/task_profile/inputs/01_L3_H6_own3.pt}
OUT=${DIAG_OUT:-$HOME/hw4-prof/gpu-diag-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}
if [[ ${1:-} != --check-tools && ! -f "$INPUT" ]]; then
  echo "Missing captured input: $INPUT"
  exit 2
fi
mkdir -p "$OUT"
source prof/find_nsight.sh
find_nsight_tools > "$OUT/tool-discovery.log" 2>&1
TOOLS_STATUS=$?
cat "$OUT/tool-discovery.log"
if [[ ${1:-} == --check-tools ]]; then exit "$TOOLS_STATUS"; fi
: > "$OUT/status.txt"
FINAL_STATUS=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HW4_ROLLOUT_GRAPH=1 HW4_FAST_ROLLOUT=1
echo "output=$OUT host=$(hostname) commit=$(git rev-parse --short HEAD)"
{
  nvidia-smi
  cat "$OUT/tool-discovery.log"
  command -v nvidia-cuda-mps-control || true
} > "$OUT/environment.txt" 2>&1

if [[ ${SCALING:-0} == 1 ]]; then
echo "1. Unprofiled worker scaling (fixed resident pool, ascending/descending order)"
timeout --kill-after=5s 100s "$PY" -u prof/gpu_diagnose.py --input "$INPUT" \
  --output "$OUT/scaling" --mode scaling --workers 8 --iters 20 \
  > "$OUT/scaling.log" 2>&1
SCALING_STATUS=$?
cat "$OUT/scaling.log"
echo "scaling_exit=$SCALING_STATUS" | tee -a "$OUT/status.txt"
if [[ "$SCALING_STATUS" != 0 ]]; then
  echo "Scaling failed; remaining profilers skipped. Inspect scaling.log."
  exit "$SCALING_STATUS"
fi
else
  echo 'scaling=skipped_by_request' | tee -a "$OUT/status.txt"
fi

if [[ "${NSYS:-1}" == 1 && -x "$NSYS_BIN" ]]; then
  echo "2. Short 8-worker timeline (timings under profiler are diagnostic only)"
  "$NSYS_BIN" profile --help > "$OUT/nsys-help.txt" 2>&1
  EXTRA=()
  if grep -q -- '--cuda-graph-trace' "$OUT/nsys-help.txt"; then
    EXTRA+=(--cuda-graph-trace=node)
  fi
  timeout --kill-after=5s 90s "$NSYS_BIN" profile --trace=cuda,nvtx,osrt --sample=none \
    --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
    --wait=all "${EXTRA[@]}" --output="$OUT/workers8" \
    "$PY" -u prof/gpu_diagnose.py --input "$INPUT" --output "$OUT/timeline" \
    --mode trace --workers 8 --iters 6 > "$OUT/nsys.log" 2>&1
  NSYS_STATUS=$?
  echo "nsys_exit=$NSYS_STATUS" | tee -a "$OUT/status.txt"
  tail -n 12 "$OUT/nsys.log"
  if [[ $NSYS_STATUS == 0 && -s "$OUT/workers8.nsys-rep" ]]; then
    timeout --kill-after=3s 25s "$NSYS_BIN" export --type sqlite --output "$OUT/workers8.sqlite" \
      "$OUT/workers8.nsys-rep" > "$OUT/nsys-export.log" 2>&1
    EXPORT_STATUS=$?
    echo "nsys_export_exit=$EXPORT_STATUS" | tee -a "$OUT/status.txt"
    if [[ $EXPORT_STATUS == 0 ]]; then
      "$PY" prof/nsys_summary.py "$OUT/workers8.sqlite" --output "$OUT/timeline-summary.json" \
        > "$OUT/timeline-summary.txt" 2>&1
      SUMMARY_STATUS=$?
      echo "timeline_summary_exit=$SUMMARY_STATUS" | tee -a "$OUT/status.txt"
      [[ $SUMMARY_STATUS == 0 ]] || FINAL_STATUS=1
      cat "$OUT/timeline-summary.txt"
    else
      FINAL_STATUS=1
      cat "$OUT/nsys-export.log"
    fi
  else
    FINAL_STATUS=1
    echo 'timeline_report=missing_or_capture_failed' | tee -a "$OUT/status.txt"
  fi
else
  if [[ ${NSYS:-1} == 0 ]]; then
    echo 'nsys=disabled' | tee -a "$OUT/status.txt"
  else
    echo 'nsys=NOT_FOUND' | tee -a "$OUT/status.txt"
    FINAL_STATUS=1
  fi
fi

if [[ "${NCU:-1}" == 1 && -x "$NCU_BIN" ]]; then
  echo "3. GEMM counters: up to two highest-total-duration names from the timeline"
  TARGETS=(1)
  [[ -s "$OUT/gemm-filter-2.txt" ]] && TARGETS+=(2)
  for INDEX in "${TARGETS[@]}"; do
    PREFIX="$OUT/gemm-$INDEX"
    FILTER='regex:.*(gemm|gemv|Gemm|Gemv|cutlass|xmma).*'
    NVTX_FILTER=()
    if [[ -s "$OUT/gemm-filter-$INDEX.txt" ]]; then
      FILTER=$(cat "$OUT/gemm-filter-$INDEX.txt")
      NVTX_FILTER=()
      echo "ncu_${INDEX}_selection=timeline_ranked_name" | tee -a "$OUT/status.txt"
    else
      echo "ncu_${INDEX}_selection=fallback_first_two_steady_eager_gemms" | tee -a "$OUT/status.txt"
    fi
    timeout --kill-after=5s 45s "$NCU_BIN" --profile-from-start off --nvtx "${NVTX_FILTER[@]}" \
    --kernel-name-base demangled --kernel-name "$FILTER" \
    --launch-count 2 --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section LaunchStats --section Occupancy --section SchedulerStats \
    --section ComputeWorkloadAnalysis --section WarpStateStats \
    --cache-control none --clock-control none --export "$PREFIX" \
    "$PY" -u prof/gpu_diagnose.py --input "$INPUT" --output "$OUT/kernel-$INDEX" \
    --mode kernel --workers 1 > "$PREFIX.log" 2>&1
    NCU_STATUS=$?
    echo "ncu_${INDEX}_exit=$NCU_STATUS" | tee -a "$OUT/status.txt"
    tail -n 20 "$PREFIX.log"
    if [[ $NCU_STATUS == 0 && -s "$PREFIX.ncu-rep" ]] && ! grep -qi 'No kernels were profiled' "$PREFIX.log"; then
      timeout 10s "$NCU_BIN" --import "$PREFIX.ncu-rep" --page details > "$PREFIX.details.txt" 2>&1
      DETAILS_STATUS=$?
      echo "ncu_${INDEX}_details_exit=$DETAILS_STATUS" | tee -a "$OUT/status.txt"
      [[ $DETAILS_STATUS == 0 ]] || FINAL_STATUS=1
    else
      echo "ncu_${INDEX}_report=missing_or_capture_failed" | tee -a "$OUT/status.txt"
      FINAL_STATUS=1
    fi
    if grep -q 'ERR_NVGPUCTRPERM' "$PREFIX.log"; then
      echo 'ncu=COUNTER_PERMISSION_DENIED; skipping further counter attempts' | tee -a "$OUT/status.txt"
      break
    fi
  done
else
  if [[ ${NCU:-1} == 0 ]]; then
    echo 'ncu=disabled' | tee -a "$OUT/status.txt"
  else
    echo 'ncu=NOT_FOUND' | tee -a "$OUT/status.txt"
    FINAL_STATUS=1
  fi
fi
echo "diagnostic_exit=$FINAL_STATUS" | tee -a "$OUT/status.txt"
echo "Existing output files in $OUT:"
find "$OUT" -maxdepth 1 -type f -printf '%f\n' | sort
exit "$FINAL_STATUS"
