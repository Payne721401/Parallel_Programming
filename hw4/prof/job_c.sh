#!/bin/bash
# Job C: verify and time the rollout CUDA graphs (model/graphs.py) against the baseline.
#   1. graph vs eager on the inputs task_profile.py saved (diffs must be exactly 0)
#   2. check.py, all six cases
#   3. evaluate.py, 8 workers, tasks 0-9 (judge-like e1)
#   4. task-by-task comparison with the baseline e1 of job A (must be identical)
# Submit from ~/Parallel_Programming/hw4:  sbatch -o ~/hw4-prof/%x-%j.log prof/job_c.sh
# Override the baseline folders with BASE_A=... INPUTS=... (sbatch --export=ALL,BASE_A=...).
#SBATCH -J hw4-c
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
BASE_A=${BASE_A:-$HOME/hw4-prof/a-452740}
INPUTS=${INPUTS:-$HOME/hw4-prof/b-452742/task_profile/inputs}
export PROF_OUT="$HOME/hw4-prof/c-${SLURM_JOB_ID:-manual}"
export HW4_ASSETS_VERIFIED=1
mkdir -p "$PROF_OUT"
echo "PROF_OUT=$PROF_OUT  host=$(hostname)  commit=$(git rev-parse --short HEAD)  start=$(date)"
echo "baseline e1: $BASE_A/eval8   captured inputs: $INPUTS"

smi_start() {
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
    --format=csv,noheader,nounits -lms 200 > "$1" &
  SMI_PID=$!
}
smi_stop() { kill "$SMI_PID" 2>/dev/null; wait "$SMI_PID" 2>/dev/null; }

echo "== 1. rollout graph vs eager on captured planner inputs =="
HW4_GRAPH_LOG=1 $PY prof/graph_check.py --inputs "$INPUTS" 2>&1

echo "== 2. check.py, all cases =="
{ time $PY check.py --output "$PROF_OUT/check.json" 2>&1 ; } 2> "$PROF_OUT/check.time"
cat "$PROF_OUT/check.time"

echo "== 3. evaluate.py, 8 workers, tasks 0-9 (judge-like) =="
smi_start "$PROF_OUT/smi_eval8.csv"
{ time HW4_GRAPH_LOG=1 $PY evaluate.py --workers 8 --tasks 0 1 2 3 4 5 6 7 8 9 --no-forward-timing \
    --output-dir "$PROF_OUT/eval8" 2>&1 ; } 2> "$PROF_OUT/eval8.time"
smi_stop
cat "$PROF_OUT/eval8.time"
echo "graph captures logged by the workers (count, line):"
grep -h '\[hw4-graph\]' "$PROF_OUT"/eval8/episodes/task-*.log | sed 's/captured in .*//' | sort | uniq -c
grep -h '\[hw4-graph\]' "$PROF_OUT"/eval8/episodes/task-*.log \
  | awk '/captured in/ { for (i = 1; i <= NF; i++) if ($i == "in") { s += $(i + 1); n++ } }
         END { if (n) printf "captures: %d, total %.2f s, mean %.3f s\n", n, s, s / n }'

echo "== 4. compare with the baseline e1 =="
$PY prof/compare_eval.py "$BASE_A/eval8" "$PROF_OUT/eval8"

echo "== 5. report =="
$PY prof/report.py "$PROF_OUT" | tee "$PROF_OUT/report.txt"
echo "done $(date)"
