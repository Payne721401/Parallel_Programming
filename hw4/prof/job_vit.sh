#!/bin/bash
# Compare HF/native startup with attention fixed at v (override ATTN_MODE if needed).
# sbatch -o ~/hw4-prof/%x-%j.log prof/job_vit.sh
# CHECKS_ONLY=1 skips the four e1 timing runs.
#SBATCH -J hw4-vit
#SBATCH -A ACD115218
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-node=1
#SBATCH -t 00:10:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="$HOME/hw4-prof/vit-${SLURM_JOB_ID:-manual}"
ARGS=()
if [[ "${CHECKS_ONLY:-0}" == "1" ]]; then
  ARGS+=(--checks-only)
fi
echo "host=$(hostname) attention=${ATTN_MODE:-v} output=$OUT"
.venv/bin/python -u prof/vit_ab.py --output "$OUT" --attention "${ATTN_MODE:-v}" "${ARGS[@]}"
