#!/usr/bin/env bash
set -euo pipefail

# One allocation, independent binaries, checked answers, no judge submissions.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
prob_dir="$(cd "$script_dir/../hw2-1" && pwd)"
cd "$prob_dir"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run inside sbatch/salloc, not on the login node.' >&2; exit 2; }
for cmd in srun mpicxx python3 hw2-1-check git; do
    command -v "$cmd" >/dev/null || { echo "Missing: $cmd" >&2; exit 2; }
done
np="${PUBLIC_NP:-8}"
repeat="${REPEAT:-3}"
tau="${TAU:-16}"
base_ref="${BASE_REF:-5c8b402}"
case_dir="${CASES_DIR:-/home/share/pp26/cases/hw2-1}"
case_list="${CASES:-01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17}"
for value in "$np" "$repeat" "$tau"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Ranks, repetitions and cutoff must be positive integers.' >&2; exit 2; }
done
((np <= 8 && np <= ${SLURM_NTASKS:-8} && tau <= 255)) || { echo 'Practice allocation supports at most 8 ranks; tau must be <=255.' >&2; exit 2; }
read -r -a case_ids <<< "$case_list"
((${#case_ids[@]})) || { echo 'Empty CASES.' >&2; exit 2; }
declare -A seen=()
for c in "${case_ids[@]}"; do
    [[ "$c" =~ ^(0[1-9]|1[0-7])$ && -z "${seen[$c]:-}" ]] || { echo "Invalid/duplicate case: $c" >&2; exit 2; }
    seen[$c]=1
    [[ -f "$case_dir/$c.txt" && -f "$case_dir/$c.golden.bin" ]] || { echo "Missing public case/golden: $case_dir/$c" >&2; exit 2; }
done
out_root="${OUT_ROOT:-$prob_dir/tmp/hw2-1-witness}"
mkdir -p "$out_root"
out="$(mktemp -d "$out_root/job-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
out="$(cd "$out" && pwd)"
mkdir "$out/src" "$out/bin" "$out/logs" "$out/inputs" "$out/results"
cp "$script_dir/summarize_hw2-1-witness.py" "$out/summarize.py"
printf 'stage\tcase\tvariant\tranks\trepeat\tmode\ttau\tstatus\tcompute_s\tthreads\tlog\n' > "$out/runs.tsv"
finish() {
    local status=$?
    trap - EXIT
    python3 "$out/summarize.py" "$out" "$repeat" "${case_ids[@]}" || status=1
    printf '%s\n' "$status" > "$out/exit-status"
    if tar --exclude='./bin' --exclude='./results' -czf "$out.tar.gz" -C "$out" .; then
        echo "Bundle: $out.tar.gz"
    else
        echo "Archive failed; original results remain at $out" >&2
        status=1
    fi
    echo "Summary: $out/summary.md"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Preserve exactly the sources that were tested, including the old non-bitmap baseline.
git show "$base_ref:hw2/hw2-1/hw2-1.cpp" > "$out/src/baseline.cpp"
cp hw2-1.cpp "$out/src/candidate.cpp"
cp main.cpp graph_gen.cpp graph_gen.hpp "$out/src/"
cp ../original_version/hw2-1.cpp "$out/src/reference.cpp"
cp "$script_dir/run_hw2-1-witness.sh" "$script_dir/batch_hw2-1-witness.sh" "$out/"
{
    date -Is
    printf 'job=%s ranks=%s repeat=%s tau=%s cases=%s baseline=%s\n' "$SLURM_JOB_ID" "$np" "$repeat" "$tau" "$case_list" "$base_ref"
    printf 'step_timeout=%s private_data_kib=1228800\n' "${STEP_TIME:-00:03:00}"
    git rev-parse HEAD
    git status --short
    mpicxx --version
    mpirun --version
    sha256sum "$out"/src/*
    srun -N1 -n1 -c1 --hint=nomultithread lscpu
} > "$out/metadata.log" 2>&1
echo "Results: $out"
flags=(-std=c++17 -O3 -march=native)
for src in main graph_gen; do
    mpicxx "${flags[@]}" -c "$out/src/$src.cpp" -o "$out/bin/$src.o"
done
for variant in baseline candidate reference phase; do
    src="$variant"
    extra=()
    if [[ "$variant" == phase ]]; then src=candidate; extra=(-DHW21_PROF -DHW21_PHASE_ONLY); fi
    mpicxx "${flags[@]}" "${extra[@]}" "$out/src/$src.cpp" "$out/bin/main.o" "$out/bin/graph_gen.o" -o "$out/bin/$variant"
done

# Remove inherited experiment knobs so each comparison changes only what its label says.
for name in "${!PP_@}"; do unset "$name"; done
ulimit -d 1228800
failed=0
run_one() {
    local stage="$1" c="$2" variant="$3" ranks="$4" rep="$5" mode="$6" cutoff="$7"
    local golden="$8" input="$9" binary="${10}"
    shift 10
    local label="$stage-$c-$variant-p$ranks-r$rep" status time threads
    local log="$out/logs/$label" output="$out/results/$label.bin"
    if env PP_WITNESS="$mode" PP_WITNESS_TAU="$cutoff" "$@" \
        srun -N1 -n"$ranks" -c1 --hint=nomultithread --kill-on-bad-exit=1 --time="${STEP_TIME:-00:03:00}" \
        "$out/bin/$binary" "$input" "$output" </dev/null > "$log.out" 2> "$log.err"; then
        if [[ ! -f "$output" ]]; then status=NO_OUTPUT
        elif hw2-1-check "$golden" "$output" > "$log.check" 2>&1; then status=PASS
        else status=WA; fi
    else status=RUNTIME_ERROR; fi
    time="$(sed -n 's/^Compute time: \([0-9.eE+-]*\) s$/\1/p' "$log.out")"
    threads="$(sed -n 's/^Threads: \([0-9-]*\).*/\1/p' "$log.out")"
    if [[ "$status" == PASS && ! "$time" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then status=NO_TIMING; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$stage" "$c" "$variant" "$ranks" "$rep" "$mode" "$cutoff" "$status" "${time:--}" "${threads:--}" "logs/$label" >> "$out/runs.tsv"
    printf '%-14s %-42s %s s\n' "$status" "$label" "${time:--}"
    if [[ "$status" != PASS ]]; then failed=$((failed + 1)); fi
}

# Full sampled vectors at small scales; repeated/isolated sources, P>N, both scramble modes,
# maximum seed/edgefactor, nsample=0, S=32, and a graph with no light edges.
printf '1 1 0 0 2 2 0 1\n' > "$out/inputs/empty.txt"
printf '8 1 11 1 256 4 0 17 255 0\n' > "$out/inputs/sparse.txt"
printf '10 16 42 0 1024 4 0 1 1023 0\n' > "$out/inputs/plain.txt"
printf '10 16 2026 1 1024 4 0 1 1023 0\n' > "$out/inputs/scrambled.txt"
printf '8 256 18446744073709551615 1 256 4 0 1 255 0\n' > "$out/inputs/dense.txt"
printf '8 16 13 0 0 32' > "$out/inputs/sources32.txt"
for ((s=0; s<32; s++)); do printf ' %s' "$s" >> "$out/inputs/sources32.txt"; done
printf '\n' >> "$out/inputs/sources32.txt"
custom=(empty sparse plain scrambled dense sources32)
custom_ranks=(1)
if ((np >= 3)); then custom_ranks+=(3); fi
if ((np != 1 && np != 3)); then custom_ranks+=("$np"); fi
for c in "${custom[@]}"; do
    input="$out/inputs/$c.txt"
    golden="$out/results/$c.golden.bin"
    srun -N1 -n1 -c1 --hint=nomultithread --kill-on-bad-exit=1 --time="${STEP_TIME:-00:03:00}" \
        "$out/bin/reference" "$input" "$golden" </dev/null > "$out/logs/reference-$c.out" 2> "$out/logs/reference-$c.err"
    for ranks in "${custom_ranks[@]}"; do
        run_one custom "$c" off "$ranks" 1 off "$tau" "$golden" "$input" candidate
        run_one custom "$c" prune "$ranks" 1 prune "$tau" "$golden" "$input" candidate
    done
done
for cutoff in 1 8 16 32 255; do
    for mode in probe prune; do
        run_one boundary dense "$mode-t$cutoff" "$np" 1 "$mode" "$cutoff" \
            "$out/results/dense.golden.bin" "$out/inputs/dense.txt" phase
    done
done
for knob in PP_BATCH=1 PP_RELABEL=0 PP_FASTGEN=0; do
    run_one boundary scrambled "$knob" "$np" 1 prune "$tau" \
        "$out/results/scrambled.golden.bin" "$out/inputs/scrambled.txt" phase "$knob"
done
if ((failed)); then echo 'Correctness gate failed; public timing cancelled.' >&2; exit 1; fi

# Rotate the order every repetition and case; never compare profiling with normal times.
variants=(baseline off prune)
ci=0
for c in "${case_ids[@]}"; do
    input="$out/inputs/$c.txt"
    cp "$case_dir/$c.txt" "$input"
    golden="$case_dir/$c.golden.bin"
    for ((rep=1; rep<=repeat; rep++)); do
        for ((i=0; i<3; i++)); do
            variant="${variants[$(((i + rep - 1 + ci) % 3))]}"
            binary=candidate; mode=off
            if [[ "$variant" == baseline ]]; then binary=baseline; fi
            if [[ "$variant" == prune ]]; then mode=prune; fi
            run_one timing "$c" "$variant" "$np" "$rep" "$mode" "$tau" "$golden" "$input" "$binary"
        done
    done
    for mode in off probe prune; do
        run_one profile "$c" "$mode" "$np" 1 "$mode" "$tau" "$golden" "$input" phase PP_WITNESS_REPORT=1
    done
    ci=$((ci + 1))
done
echo "Completed; failures=$failed"
((failed == 0))
