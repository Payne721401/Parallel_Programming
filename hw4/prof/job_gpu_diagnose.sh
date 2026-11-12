#!/bin/bash
# One allocation, no full episodes. Each tool has a wall-time cap.
# Optional: NSYS=0 skips the timeline; NCU=0 skips hardware counters.
#SBATCH -J hw4-gpu-diag
#SBATCH -A ACD115218
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-node=1
#SBATCH -t 00:05:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PY=.venv/bin/python
INPUT=${INPUT:-$HOME/hw4-prof/b-452742/task_profile/inputs/01_L3_H6_own3.pt}
OUT=${DIAG_OUT:-$HOME/hw4-prof/gpu-diag-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}
if [[ ! -f "$INPUT" ]]; then
  echo "Missing captured input: $INPUT"
  exit 2
fi
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HW4_ROLLOUT_GRAPH=1 HW4_FAST_ROLLOUT=1
echo "output=$OUT host=$(hostname) commit=$(git rev-parse --short HEAD)"
{
  nvidia-smi
  command -v nsys ncu nvidia-cuda-mps-control || true
  if command -v nsys >/dev/null; then nsys --version; fi
  if command -v ncu >/dev/null; then ncu --version; fi
} > "$OUT/environment.txt" 2>&1

echo "1. Unprofiled worker scaling (fixed resident pool, ascending/descending order)"
timeout --kill-after=5s 100s "$PY" -u prof/gpu_diagnose.py --input "$INPUT" \
  --output "$OUT/scaling" --mode scaling --workers 8 --iters 20 \
  > "$OUT/scaling.log" 2>&1
SCALING_STATUS=$?
cat "$OUT/scaling.log"
echo "scaling_exit=$SCALING_STATUS" | tee "$OUT/status.txt"
if [[ "$SCALING_STATUS" != 0 ]]; then
  echo "Scaling failed; remaining profilers skipped. Inspect scaling.log."
  exit "$SCALING_STATUS"
fi

if [[ "${NSYS:-1}" == 1 ]] && command -v nsys >/dev/null; then
  echo "2. Short 8-worker timeline (timings under profiler are diagnostic only)"
  nsys profile --help > "$OUT/nsys-help.txt" 2>&1
  EXTRA=()
  if grep -q -- '--cuda-graph-trace' "$OUT/nsys-help.txt"; then
    EXTRA+=(--cuda-graph-trace=node)
  fi
  timeout --kill-after=5s 90s nsys profile --trace=cuda,nvtx,osrt --sample=none \
    --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
    --wait=all "${EXTRA[@]}" --output="$OUT/workers8" \
    "$PY" -u prof/gpu_diagnose.py --input "$INPUT" --output "$OUT/timeline" \
    --mode trace --workers 8 --iters 6 > "$OUT/nsys.log" 2>&1
  echo "nsys_exit=$?" | tee -a "$OUT/status.txt"
  tail -n 12 "$OUT/nsys.log"
else
  echo "nsys=disabled_or_unavailable" | tee -a "$OUT/status.txt"
fi

if [[ "${NCU:-1}" == 1 ]] && command -v ncu >/dev/null; then
  echo "3. Two backward GEMM kernels, one process (not a global bottleneck verdict)"
  timeout --kill-after=5s 60s ncu --profile-from-start off --nvtx \
    --nvtx-include 'diag_backward/' --kernel-name 'regex:.*(gemm|gemv|Gemm|Gemv|cutlass|xmma).*' \
    --launch-count 2 --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section LaunchStats --section Occupancy --section SchedulerStats \
    --section ComputeWorkloadAnalysis --section WarpStateStats \
    --cache-control none --clock-control none --export "$OUT/gemm" \
    "$PY" -u prof/gpu_diagnose.py --input "$INPUT" --output "$OUT/kernel" \
    --mode kernel --workers 1 > "$OUT/ncu.log" 2>&1
  echo "ncu_exit=$?" | tee -a "$OUT/status.txt"
  tail -n 20 "$OUT/ncu.log"
else
  echo "ncu=disabled_or_unavailable" | tee -a "$OUT/status.txt"
fi
echo "Reports: $OUT; scaling/summary.json, workers8.nsys-rep, gemm.ncu-rep"
echo "Nsight failures or permission errors remain in nsys.log/ncu.log; no privileges or MPS configuration changed."
