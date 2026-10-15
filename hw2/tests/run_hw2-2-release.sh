#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../hw2-2"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "submit through ../tests/batch_hw2-2.sh, not on the login node" >&2
    exit 1
fi
root="${OUT_ROOT:-tmp/hw2-2-release-$SLURM_JOB_ID}"
mkdir -p "$root/bin"
unset PP_HOT PP_HOT_MB PP_HOT_REPORT PP_ACCUM PP_HOT_DENSE CASES SHAPE CUSTOM_CASES
export TIMING=0 VERBOSE=1
echo "== submission-default validation: $root =="
git log -1 --oneline
make -B hw2-2
mpicxx -std=c++17 -O2 -g -D_GLIBCXX_ASSERTIONS -fsanitize=undefined \
    -fno-sanitize-recover=all "$script_dir/test_hw2-2-boundary.cpp" -o "$root/bin/boundary"
srun -N2 -n2 --ntasks-per-node=1 -c1 --hint=nomultithread "$root/bin/boundary" \
    </dev/null >"$root/boundary.log" 2>&1 || { cat "$root/boundary.log"; exit 1; }
cat "$root/boundary.log"

# Actual official c04 through the fixed driver, with fresh output paths on each run.
for rep in 1 2 3; do
    CASES=c04 SHAPE="-N2 -n2 --ntasks-per-node=1" OUT_ROOT="$root/c04-$rep" \
        bash "$script_dir/run_hw2-2.sh" public
done
# Keep this root separate from custom data: ranks runs the 14 public cases only.
OUT_ROOT="$root/ranks" bash "$script_dir/run_hw2-2.sh" ranks
OUT_ROOT="$root/custom" bash "$script_dir/run_hw2-2.sh" custom
echo "== release checks complete: 120 driver/checker runs plus in-memory boundary checks =="
