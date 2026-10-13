#!/bin/bash -l
#SBATCH --job-name=hw2-1-witness
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=40
#SBATCH --output=hw2-1-witness-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from hw2/hw2-1}"
[[ -f hw2-1.cpp && -f ../tests/run_hw2-1-witness.sh ]] || { echo 'Submit from hw2/hw2-1.' >&2; exit 2; }
if type module >/dev/null 2>&1; then
    module load gcc/15.2.0 openmpi/5.0.10
fi
bash ../tests/run_hw2-1-witness.sh
