#!/bin/bash
# Profiling job B: in-process task profile (+ torch.profiler windows, captured inputs),
# microbench eager vs CUDA graph, and the same with 8 processes sharing the GPU.
# Submit from ~/hw4:  sbatch -o ~/hw4-prof/%x-%j.log prof/job_b.sh
#SBATCH -J hw4prof-b
#SBATCH -A ACD115218
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-node=1
#SBATCH -t 00:10:00
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PY=.venv/bin/python
export PROF_OUT="$HOME/hw4-prof/b-${SLURM_JOB_ID:-manual}"
export HW4_ASSETS_VERIFIED=1
mkdir -p "$PROF_OUT"
echo "PROF_OUT=$PROF_OUT  host=$(hostname)  start=$(date)"

echo "== 1. task profile, task 0, 1 worker =="
$PY prof/task_profile.py --tasks 0 2>&1

echo "== 2. microbench, 1 process =="
$PY prof/microbench.py 2>&1

echo "== 3. microbench, 8 processes at once =="
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
  --format=csv,noheader,nounits -lms 200 > "$PROF_OUT/smi_mb8.csv" &
SMI_PID=$!
$PY prof/microbench.py --procs 8 --iters 30 --graph-iters 100 2>&1
kill "$SMI_PID" 2>/dev/null; wait "$SMI_PID" 2>/dev/null

echo "== summary =="
$PY prof/report.py "$PROF_OUT" | tee "$PROF_OUT/report.txt"
echo "done $(date)"
