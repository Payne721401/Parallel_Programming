#!/bin/bash
# Stage 1: sbatch --export=ALL,ATTN_MODE=sdpa,BASE_MODE=off -o ~/hw4-prof/%x-%j.log prof/job_attention.sh
# Stage 2: sbatch --export=ALL,ATTN_MODE=v,BASE_MODE=sdpa -o ~/hw4-prof/%x-%j.log prof/job_attention.sh
#SBATCH -J hw4-attn
#SBATCH -A ACD115218
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-node=1
#SBATCH -t 00:10:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ATTN_MODE=${ATTN_MODE:-sdpa}
BASE_MODE=${BASE_MODE:-off}
INPUTS=${INPUTS:-$HOME/hw4-prof/b-452742/task_profile/inputs}
OUT="$HOME/hw4-prof/attn-${ATTN_MODE}-${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) candidate=$ATTN_MODE baseline=$BASE_MODE output=$OUT"
.venv/bin/python -u prof/attention_ab.py --mode "$ATTN_MODE" --baseline "$BASE_MODE" \
    --inputs "$INPUTS" --output "$OUT"
