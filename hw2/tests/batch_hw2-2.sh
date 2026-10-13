#!/bin/bash -l
#SBATCH --job-name=hw2-2-tests
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=40
#SBATCH --output=hw2-2-batch.log
#SBATCH --open-mode=truncate

# HW 2-2 tests as a batch job: Slurm runs it on the compute nodes whether or not the ssh session
# that submitted it survives. Two nodes, 4 ranks each, 16G per node: the practice QoS's limit.
# Submit from hw2/hw2-2:
#
#   sbatch ../tests/batch_hw2-2.sh            # the default steps below
#   sbatch ../tests/batch_hw2-2.sh 'bash ../tests/run_hw2-2.sh public' \
#                                  'TIMING=1 CASES="p01 p02 p03" bash ../tests/run_hw2-2.sh public'
#
# Each argument is one step: a command line run from hw2/hw2-2, in order, every step even after
# one fails. The log is hw2-2-batch.log in hw2/hw2-2 (sbatch -o <file> for another), rewritten by
# each job; it ends with "== done" and each step's status and duration.

cd "${SLURM_SUBMIT_DIR:-.}" || exit 2
if [[ ! -f hw2-2.cpp || ! -f ../tests/run_hw2-2.sh ]]; then
    echo "submit from hw2/hw2-2 (the job runs where it was submitted)" >&2
    exit 2
fi
if type module >/dev/null 2>&1; then
    module load gcc/15.2.0 openmpi/5.0.10
fi

steps=("$@")
if ((${#steps[@]} == 0)); then
    steps=(
        'bash ../tests/run_hw2-2.sh custom'
        'bash ../tests/run_hw2-2.sh ranks'
        'bash ../tests/run_hw2-2.sh public'
        'TIMING=1 CASES="p01 p02 p03" bash ../tests/run_hw2-2.sh public'
    )
fi

echo "== job ${SLURM_JOB_ID:-?} on $(hostname), $(date '+%F %T'), at $(git log -1 --format='%h %s' | cut -c1-70)"
summary=()
for s in "${steps[@]}"; do
    echo
    echo "##### $s"
    start=$SECONDS
    if eval "$s"; then st=ok; else st="FAILED (exit $?)"; fi
    summary+=("$(printf '%-16s %5d s  %s' "$st" $((SECONDS - start)) "$s")")
done
echo
echo "== done $(date '+%F %T')"
printf '  %s\n' "${summary[@]}"
