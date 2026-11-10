#!/bin/bash
# Profiling job A: startup breakdown, real check.py, import times, evaluate.py with 8 and 1 workers.
# Submit from ~/hw4:  sbatch -o ~/hw4-prof/%x-%j.log prof/job_a.sh
#SBATCH -J hw4prof-a
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
export PROF_OUT="$HOME/hw4-prof/a-${SLURM_JOB_ID:-manual}"
export HW4_ASSETS_VERIFIED=1
mkdir -p "$PROF_OUT"
echo "PROF_OUT=$PROF_OUT  host=$(hostname)  start=$(date)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,memory.total --format=csv | tee "$PROF_OUT/gpu.txt"

smi_start() {
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
    --format=csv,noheader,nounits -lms 200 > "$1" &
  SMI_PID=$!
}
smi_stop() { kill "$SMI_PID" 2>/dev/null; wait "$SMI_PID" 2>/dev/null; }

echo "== 1. startup breakdown (cold, then warm page cache) =="
for i in 1 2; do
  T0=$(date +%s.%N)
  PROF_T0=$T0 PROF_TAG=run$i $PY prof/startup.py 2>&1
  T1=$(date +%s.%N)
  echo "$T0 $T1" > "$PROF_OUT/startup_wall_run$i.txt"
  awk -v a="$T0" -v b="$T1" 'BEGIN { printf "startup run: process wall %.3f s\n", b - a }'
done

echo "== 2. real check.py on one case =="
CASE=$(head -n1 "$PROF_OUT/case_ids.txt" 2>/dev/null)
{ time $PY check.py ${CASE:+--case "$CASE"} --output "$PROF_OUT/check_real.json" 2>&1 ; } 2> "$PROF_OUT/check_real.time"
cat "$PROF_OUT/check_real.time"

echo "== 3. import times =="
$PY -X importtime -c "import torch, transformers; from transformers import ViTConfig, ViTModel" 2> "$PROF_OUT/importtime.txt"

echo "== 4. evaluate.py, 8 workers, tasks 0-9 (judge-like) =="
smi_start "$PROF_OUT/smi_eval8.csv"
{ time $PY evaluate.py --workers 8 --tasks 0 1 2 3 4 5 6 7 8 9 --no-forward-timing \
    --output-dir "$PROF_OUT/eval8" 2>&1 ; } 2> "$PROF_OUT/eval8.time"
smi_stop
cat "$PROF_OUT/eval8.time"

echo "== 5. evaluate.py, 1 worker, tasks 0 1 (no contention) =="
smi_start "$PROF_OUT/smi_eval1.csv"
{ time $PY evaluate.py --workers 1 --tasks 0 1 --no-forward-timing \
    --output-dir "$PROF_OUT/eval1" 2>&1 ; } 2> "$PROF_OUT/eval1.time"
smi_stop
cat "$PROF_OUT/eval1.time"

echo "== summary =="
$PY prof/report.py "$PROF_OUT" | tee "$PROF_OUT/report.txt"
echo "done $(date)"
