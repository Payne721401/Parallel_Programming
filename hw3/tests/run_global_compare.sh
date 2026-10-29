#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-}
case "$mode" in
  build)
    shift
    exec python3 -u tests/global_compare.py build "$@"
    ;;
  compare)
    shift
    [[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Compare requires an existing srun/sbatch allocation.' >&2; exit 2; }
    logdir="tmp/global-compare-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
    mkdir -p "$logdir"
    exec > >(tee "$logdir/session.log") 2>&1
    echo "Logs: $logdir"
    # Python creates its result directory exclusively; the tee directory is its parent.
    python3 -u tests/global_compare.py compare --outdir "$logdir/results" "$@"
    ;;
  *)
    echo 'Build without GPU: bash tests/run_global_compare.sh build --build-dir tmp/global-build-v1' >&2
    echo 'Inside Slurm: bash tests/run_global_compare.sh compare --build-dir tmp/global-build-v1 [--method combined|multistate|rolling|sparse-bits] [--group target|large|sparse|multistate|sparse-bits | --cases 04-07,09-12]' >&2
    exit 2
    ;;
esac
