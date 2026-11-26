#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Build on the login node; keep binaries, outputs and logs together for this allocation.
dir=$(mktemp -d "$PWD/measure-experts-XXXXXX")
echo "Results: $dir"
git rev-parse HEAD > "$dir/candidate.txt"
git show 3f2a2df:hw5/hw5.cu > "$dir/hw5.cu"
cp main.cu mixtral.hpp Makefile "$dir/"
make -C "$dir" hw5
make hw5 hw5_serial_experts
cp hw5 "$dir/candidate"
cp hw5_serial_experts "$dir/serial"
export dir
srun -A ACD115218 -p dev -N1 -n1 -c4 --gpus-per-node=2 -t 00:06:00 \
    bash -ec '
        run_cases() {
            local binary=$1 c
            shift
            local args=()
            for c in "$@"; do args+=("cases/$c.txt" "$dir/$binary-$c.bin"); done
            echo "=== $binary ==="
            "$dir/$binary" weights.bin "${args[@]}"
        }
        run_cases hw5 03 09 12
        run_cases candidate 12 09 03
        run_cases serial 09 03 12
    ' 2>&1 | tee "$dir/timing.log"
(
    for c in 03 09 12; do
        for binary in hw5 candidate serial; do
            echo "=== checker $binary case $c ==="
            hw5-check "goldens/$c.golden.bin" "$dir/$binary-$c.bin" || exit 1
        done
        cmp "$dir/candidate-$c.bin" "$dir/serial-$c.bin" || exit 1
        echo "case $c: concurrent and serial outputs byte-identical"
    done
) 2>&1 | tee "$dir/check.log"
