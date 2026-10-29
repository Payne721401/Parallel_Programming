#!/usr/bin/env bash
# User-run GPU experiment only; this script never requests an allocation.
set -euo pipefail
cd "$(dirname "$0")/.."
build_dir=
cases=01-04,08,10-12
while (($#)); do
  case "$1" in
    --build-dir)
      (($# >= 2)) || { echo '--build-dir requires a path.' >&2; exit 2; }
      build_dir=$2; shift 2 ;;
    --cases)
      (($# >= 2)) || { echo '--cases requires case IDs/ranges.' >&2; exit 2; }
      cases=$2; shift 2 ;;
    --help|-h)
      echo 'Inside Slurm: bash tests/run_regression_control.sh --build-dir tmp/regression-build-v1'
      echo 'Optional: --cases 01-04,08,10-12 (default); case 11 is always included.'
      echo 'A/B: original binary vs current binary, new flags disabled.'
      echo 'B/C: exact same current binary, disabled vs multistate only.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Requires an existing srun/sbatch allocation.' >&2; exit 2; }
[[ -n $build_dir ]] || { echo '--build-dir is required.' >&2; exit 2; }
[[ -f $build_dir/BUILT ]] || { echo "Incomplete frozen build: $build_dir" >&2; exit 2; }
# Verify the original source identity and every frozen hash before any GPU work.
python3 - "$build_dir" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "tests")
from global_compare import require_original_baseline, verify_manifest
try:
    manifest, _ = verify_manifest(Path(sys.argv[1]).resolve())
    require_original_baseline(manifest)
except (OSError, ValueError, KeyError) as error:
    raise SystemExit("Regression preflight failed: " + str(error))
print("Original baseline verified: " + manifest["baseline_revision"], flush=True)
PY
command -v timeout >/dev/null || { echo 'GNU timeout is required for the total wall-time cap.' >&2; exit 2; }
logdir="tmp/regression-control-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Regression control logs: $logdir"
echo "Same Slurm allocation: $SLURM_JOB_ID; cases: $cases"
SECONDS=0
run_phase() {
  local label=$1 method=$2 baseline_binary=$3
  local hard_limit=$((420 - SECONDS - 10))
  ((hard_limit > 200)) && hard_limit=200
  ((hard_limit > 40)) || { echo 'Total 420-second budget exhausted; no COMPLETE.' >&2; return 1; }
  local compare_budget=$((hard_limit - 10))
  echo "$label"
  # Two phases each have at most 190s of compare work and a 200s hard cap.
  # The remaining-time calculation and reserve also bound partial/failing runs.
  timeout --signal=TERM --kill-after=3s "${hard_limit}s" \
    bash tests/run_global_compare.sh compare --build-dir "$build_dir" \
      --method "$method" --baseline-method disabled --baseline-binary "$baseline_binary" \
      --cases "$cases" --blocks 2 --profile-blocks 1 --budget-seconds "$compare_budget"
}
run_phase 'A/B: frozen original binary vs current binary; all six new flags disabled.' disabled baseline
run_phase 'B/C: same frozen current binary; disabled vs multistate only.' multistate candidate
((SECONDS < 420)) || { echo 'Total 420-second budget exhausted; no COMPLETE.' >&2; exit 1; }
printf 'Both comparisons completed in the same allocation.\n' > "$logdir/COMPLETE"
echo "COMPLETE: $logdir/COMPLETE"
