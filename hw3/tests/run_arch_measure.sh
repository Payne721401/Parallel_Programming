#!/usr/bin/env bash
# Read-only baseline diagnostics: no kernel or production defaults are changed.
set -euo pipefail
cd "$(dirname "$0")/.."
group=${1:-}; mode=${2:-all}
case "$group" in sparse|r1|large) ;; *) echo 'usage: bash tests/run_arch_measure.sh sparse|r1|large [all|timing|ncu|plan]' >&2; exit 2 ;; esac
case "$mode" in all|timing|ncu|plan) ;; *) echo 'invalid mode' >&2; exit 2 ;; esac
[[ $# -le 2 ]] || { echo 'too many arguments' >&2; exit 2; }
if [[ $mode == plan ]]; then
    exec python3 tests/arch_measure.py "$group" --mode plan
fi
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run through srun/sbatch, not on the login node.' >&2; exit 2; }
command -v hw3-1-check >/dev/null || { echo 'Missing hw3-1-check' >&2; exit 2; }
[[ -x hw3-1 ]] || { echo 'Run make hw3-1 first.' >&2; exit 2; }
if [[ $mode != timing ]]; then
    command -v ncu >/dev/null || { echo 'Missing ncu; load cuda/12.4 before submission.' >&2; exit 2; }
fi
logdir="tmp/arch-${group}-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
python3 -u tests/arch_measure.py "$group" --mode "$mode" --outdir "$logdir"
