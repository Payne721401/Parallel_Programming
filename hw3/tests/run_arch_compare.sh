#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
method=${1:-}; mode=${2:-compare}
case "$method" in reduce|r1|r1shared|large|all) ;; *) echo 'usage: bash tests/run_arch_compare.sh reduce|r1|r1shared|large|all [compare|check]' >&2; exit 2 ;; esac
case "$mode" in compare|check) ;; *) echo 'invalid mode' >&2; exit 2 ;; esac
[[ $# -le 2 && -n ${SLURM_JOB_ID:-} ]] || { echo 'Run through srun/sbatch.' >&2; exit 2; }
[[ -x hw3-1 && -x tests/test_hw3-1 ]] || { echo 'Run make hw3-1 tests/test_hw3-1 first.' >&2; exit 2; }
command -v hw3-1-check >/dev/null || { echo 'Missing hw3-1-check' >&2; exit 2; }
logdir="tmp/arch-compare-${method}-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
python3 -u tests/arch_compare.py "$method" --mode "$mode" --outdir "$logdir"
