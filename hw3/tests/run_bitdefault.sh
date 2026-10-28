#!/usr/bin/env bash
# Short release gate: exact compact kernel + automatic default/fallback dispatch.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run through srun/sbatch.' >&2; exit 2; }
unset PP_KERNEL PP_ACTIVE PP_R1 PP_MOORE PP_SUPPORT PP_EVOLVE_EVERY
unset PP_BITPLANE PP_BIT_ROWS PP_BIT_LOGIC
export PP_PROFILE=1 PP_GRAPH=0 PP_EVOLVE=0
logdir="tmp/bitdefault-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$logdir"
exec > >(tee "$logdir/session.log") 2>&1
echo "Logs: $logdir"
echo "Revision: $(git rev-parse HEAD)"
sha256sum hw3-1 tests/test_hw3-1 hw3-1.cu r1_bits.hpp r1_bitplane.cuh
echo 'Forced bit-plane, default compact/Rows4: 452 CPU comparisons'
PP_ACTIVE=0 PP_BITPLANE=1 timeout 90s ./tests/test_hw3-1 --bitplane > "$logdir/compact.log" 2>&1 || { cat "$logdir/compact.log"; exit 1; }
python3 tests/summarize_bitlogic.py gate "$logdir/compact.log" 4 compact dense
echo 'Automatic small-grid fallback: 90 CPU comparisons'
timeout 90s ./tests/test_hw3-1 --dense-r1 > "$logdir/default.log" 2>&1 || { cat "$logdir/default.log"; exit 1; }
grep -q 'PASS 90/90 ' "$logdir/default.log"
echo 'PASS 90/90 default dense-R1 CPU comparisons'
bash tests/inspect_bitplane.sh cases 01 02 03 04 05 06 07 08 09 10 11 12
echo 'Default dispatch gate complete. Run hw3-1-judge without PP_* overrides for official timing.'
