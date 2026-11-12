#!/bin/bash
# Run evaluate.py (8 workers) on any set of tasks, e.g. the 50-task reference in two halves:
#   sbatch --export=ALL,TASKS="$(seq -s ' ' 0 24)",TAG=t00-24  -o ~/hw4-prof/%x-%j.log prof/job_tasks.sh
#   sbatch --export=ALL,TASKS="$(seq -s ' ' 25 49)",TAG=t25-49 -o ~/hw4-prof/%x-%j.log prof/job_tasks.sh
# Output: ~/hw4-prof/tasks-<TAG>-<jobid>/eval8 (episodes/task-NN.json, report.json).
#SBATCH -J hw4-tasks
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
TASKS=${TASKS:-0 1 2 3 4 5 6 7 8 9}
TAG=${TAG:-run}
export PROF_OUT="$HOME/hw4-prof/tasks-${TAG}-${SLURM_JOB_ID:-manual}"
export HW4_ASSETS_VERIFIED=1
mkdir -p "$PROF_OUT"
echo "PROF_OUT=$PROF_OUT  host=$(hostname)  commit=$(git rev-parse --short HEAD)  tasks=$TASKS  start=$(date)"
echo "== check.py: one case in a fresh process (startup time), then all cases =="
{ time $PY check.py --case 01 --output "$PROF_OUT/check01.json" 2>&1 ; } 2> "$PROF_OUT/check01.time"
cat "$PROF_OUT/check01.time"
$PY check.py --output "$PROF_OUT/check.json" 2>&1
$PY -c "import sys; sys.path.insert(0, '.'); import runner, torch; runner.load_model(); print('transformers imported:', 'transformers' in sys.modules)"
echo "== evaluate.py =="
# shellcheck disable=SC2086
{ time $PY evaluate.py --workers 8 --tasks $TASKS --no-forward-timing \
    --output-dir "$PROF_OUT/eval8" 2>&1 ; } 2> "$PROF_OUT/eval8.time"
cat "$PROF_OUT/eval8.time"
$PY prof/report.py "$PROF_OUT" | tee "$PROF_OUT/report.txt"
echo "done $(date)"
