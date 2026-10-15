#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../hw2-2"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "submit through ../tests/batch_hw2-2.sh; do not run on the login node" >&2
    exit 1
fi
root="${OUT_ROOT:-tmp/hw2-2-dense-$SLURM_JOB_ID}"
cases="${CASES:-p01 p02 p03}"
ref="${AB_REF:-d5b33a4}"
mkdir -p "$root/bin"
export PP_HOT=1 PP_HOT_MB=256 PP_ACCUM=auto PP_HOT_DENSE=1 PP_HOT_REPORT=0 TIMING=0
echo "== dense-hot experiment: baseline=$ref auto; candidate=auto+dense; results=$root =="

mpicxx -std=c++17 -O3 -march=native "$script_dir/test_hw2-2-accum.cpp" -o "$root/bin/test-accum"
(ulimit -d 1228800; srun -N1 -n1 -c1 --hint=nomultithread "$root/bin/test-accum" </dev/null)

# Capture the cluster compiler's actual vectorization decisions without changing timed builds.
mpicxx -std=c++17 -O3 -march=native -fopt-info-vec-optimized="$root/vectorization.txt" \
    -c hw2-2.cpp -o "$root/bin/vector-check.o"
echo "== dense loop source location =="
grep -n 'dst\[j\] += scale' hw2-2.cpp

CUSTOM_CASES="" OUT_ROOT="$root/check" bash "$script_dir/run_hw2-2.sh" custom
CASES="c01 c02 c03 c04 c05 c06 c07 c08 c09 c10 c11 c12 c13 c14" \
    OUT_ROOT="$root/check" bash "$script_dir/run_hw2-2.sh" public

# d5b33a4 understands auto but ignores PP_HOT_DENSE. Only normal builds count as A/B times.
AB_REF="$ref" CASES="$cases" REPEAT="${REPEAT:-3}" OUT_ROOT="$root/ab" \
    bash "$script_dir/run_hw2-2.sh" ab

for dense in 0 1; do
    PP_HOT_DENSE="$dense" PP_HOT_REPORT=1 TIMING=1 CASES="$cases" \
        OUT_ROOT="$root/profile-$dense" bash "$script_dir/run_hw2-2.sh" public
done
echo "== dense-hot experiment complete: $root =="
