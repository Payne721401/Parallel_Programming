#!/bin/bash -l
#SBATCH --job-name=hw2-1-tests
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=40
#SBATCH --output=hw2-1-batch.log
#SBATCH --open-mode=truncate

# HW 2-1 tests as a batch job: Slurm runs it on the compute node whether or not the ssh session
# that submitted it survives. Submit from hw2/hw2-1:
#
#   sbatch ../tests/batch_hw2-1.sh            # the default steps below
#   sbatch ../tests/batch_hw2-1.sh 'bash ../tests/run_hw2-1.sh custom' \
#                                  'TIMING=1 CASES="07" bash ../tests/run_hw2-1.sh public'
#
# Each argument is one step: a command line run from hw2/hw2-1, in order, every step even after
# one fails. The log is hw2-1-batch.log in hw2/hw2-1 (sbatch -o <file> for another), rewritten by
# each job. Watch it with `tail -f hw2-1-batch.log`; `squeue -u $USER` shows whether the job is
# still queued or running. The log ends with "== done" and each step's status and duration.
# The resources above are the practice QoS's (8 ranks, 32G); -t on the sbatch line overrides the
# time limit.

cd "${SLURM_SUBMIT_DIR:-.}" || exit 2
if [[ ! -f hw2-1.cpp || ! -f ../tests/run_hw2-1.sh ]]; then
    echo "submit from hw2/hw2-1 (the job runs where it was submitted)" >&2
    exit 2
fi
if type module >/dev/null 2>&1; then
    module load gcc/15.2.0 openmpi/5.0.10
fi

steps=("$@")
if ((${#steps[@]} == 0)); then
    steps=(
        'bash ../tests/run_hw2-1.sh custom'
        'bash ../tests/run_hw2-1.sh knobs'
        'CASES="05 07 08 13" bash ../tests/run_hw2-1.sh ab'
        'TIMING=1 CASES="07 09 10" bash ../tests/run_hw2-1.sh public'
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
