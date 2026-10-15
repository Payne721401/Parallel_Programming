#!/usr/bin/env bash
set -euo pipefail

# Independent source snapshots/binaries; safe to submit bucket and sample jobs separately.
mode="${1:-all}"
[[ "$mode" == all || "$mode" == bucket || "$mode" == sample ]] || {
    echo "usage: $0 [all|bucket|sample]" >&2; exit 2;
}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../hw2-1"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run inside sbatch/salloc, not on the login node.' >&2; exit 2; }
for cmd in srun mpicxx python3 hw2-1-check; do command -v "$cmd" >/dev/null; done
np="${PUBLIC_NP:-8}"
repeat="${REPEAT:-3}"
case_dir="${CASES_DIR:-/home/share/pp26/cases/hw2-1}"
read -r -a cases <<< "${CASES:-07 10 12 13}"
[[ "$np" =~ ^[1-8]$ && "$repeat" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid ranks/repetitions.' >&2; exit 2; }
((np <= ${SLURM_NTASKS:-8} && ${#cases[@]} > 0)) || exit 2
declare -A seen=()
for c in "${cases[@]}"; do
    [[ "$c" =~ ^(0[1-9]|1[0-7])$ && -z "${seen[$c]:-}" ]] || exit 2
    seen[$c]=1
    [[ -f "$case_dir/$c.txt" && -f "$case_dir/$c.golden.bin" ]] || {
        echo "Missing case/golden: $case_dir/$c" >&2; exit 2;
    }
done
root="${OUT_ROOT:-tmp/hw2-1-search}"
mkdir -p "$root"
out="$(mktemp -d "$root/$mode-${SLURM_JOB_ID}-XXXXXX")"
out="$(cd "$out" && pwd)"
mkdir "$out/src" "$out/bin" "$out/logs" "$out/inputs" "$out/results"
cp hw2-1.cpp main.cpp graph_gen.cpp graph_gen.hpp "$out/src/"
cp ../original_version/hw2-1.cpp "$out/src/reference.cpp"
cp "$script_dir/summarize_hw2-1-search.py" "$out/"
cp "$script_dir/run_hw2-1-search.sh" "$out/"
printf 'stage\tcase\tvariant\tranks\trepeat\tdivisor\tstatus\tcompute_s\tlog\n' > "$out/runs.tsv"
completed=0
finish() {
    local status=$?
    trap - EXIT
    ((completed)) || status=1
    printf '%s\n' "$status" > "$out/exit-status"
    python3 "$out/summarize_hw2-1-search.py" "$out" "$mode" "$repeat" "${cases[@]}" || status=1
    printf '%s\n' "$status" > "$out/exit-status"
    echo "Results: $out"
    echo "Summary: $out/summary.md"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
{
    date -Is
    printf 'mode=%s job=%s ranks=%s repeat=%s cases=%s\n' "$mode" "$SLURM_JOB_ID" "$np" "$repeat" "${cases[*]}"
    echo 'witness=prune tau=16 light=1; inherited PP_* variables cleared'
    git rev-parse HEAD
    git status --short
    mpicxx --version
    sha256sum "$out"/src/*
    srun -N1 -n1 -c1 --hint=nomultithread lscpu
} > "$out/metadata.log" 2>&1
for name in "${!PP_@}"; do unset "$name"; done
echo "Results: $out"
flags=(-std=c++17 -O3 -march=native)
for src in main graph_gen; do
    mpicxx "${flags[@]}" -c "$out/src/$src.cpp" -o "$out/bin/$src.o"
done
builds=(normal phase reference)
[[ "$mode" == sample ]] || builds+=(bucket atomic)
for binary in "${builds[@]}"; do
    src=hw2-1
    extra=()
    [[ "$binary" != reference ]] || src=reference
    case "$binary" in
        phase) extra=(-DHW21_PROF -DHW21_PHASE_ONLY) ;;
        bucket) extra=(-DHW21_PROF -DHW21_PHASE_ONLY -DHW21_BUCKET_PROF) ;;
        atomic) extra=(-DHW21_PROF -DHW21_PHASE_ONLY -DHW21_BUCKET_PROF -DHW21_ATOMIC_PROF) ;;
    esac
    mpicxx "${flags[@]}" "${extra[@]}" "$out/src/$src.cpp" "$out/bin/main.o" "$out/bin/graph_gen.o" -o "$out/bin/$binary"
done
ulimit -d 1228800
passed=0
failed=0
run_one() {
    local stage="$1" c="$2" binary="$3" ranks="$4" rep="$5" divisor="$6" input="$7" golden="$8"
    shift 8
    local label="$stage-$c-$binary-d$divisor-p$ranks-r$rep" status time
    local log="$out/logs/$label" output="$out/results/$label.bin"
    if env PP_WITNESS=prune PP_WITNESS_TAU=16 PP_WITNESS_LIGHT=1 \
        PP_WITNESS_SAMPLE="$divisor" PP_WITNESS_REPORT=1 "$@" \
        srun -N1 -n"$ranks" -c1 --hint=nomultithread --kill-on-bad-exit=1 --time="${STEP_TIME:-00:05:00}" \
        "$out/bin/$binary" "$input" "$output" </dev/null > "$log.out" 2> "$log.err"; then
        if [[ ! -f "$output" ]]; then status=NO_OUTPUT
        elif hw2-1-check "$golden" "$output" > "$log.check" 2>&1; then status=PASS
        else status=WA; fi
    else status=RUNTIME_ERROR; fi
    time="$(sed -n 's/^Compute time: \([0-9.eE+-]*\) s$/\1/p' "$log.out")"
    if [[ "$status" == PASS && ! "$time" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then status=NO_TIMING; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$stage" "$c" "$binary" "$ranks" "$rep" "$divisor" "$status" "${time:--}" "logs/$label" >> "$out/runs.tsv"
    printf '%-14s %-48s %s s\n' "$status" "$label" "${time:--}"
    if [[ "$status" == PASS ]]; then passed=$((passed + 1)); else failed=$((failed + 1)); fi
}

# Full distance vectors on small cases, including empty witness graphs, P>N, repeated
# sources, both scrambles, and the largest seed/edgefactor. No public timing after a failure.
printf '1 1 0 0 2 3 0 1 0\n' > "$out/inputs/empty.txt"
printf '8 1 11 1 256 4 0 17 255 0\n' > "$out/inputs/sparse.txt"
printf '10 16 42 0 1024 4 0 1 1023 0\n' > "$out/inputs/plain.txt"
printf '8 256 18446744073709551615 1 256 4 0 1 255 0\n' > "$out/inputs/dense.txt"
divisors=(1)
[[ "$mode" == bucket ]] || divisors+=(2 4)
ranks=(1)
((np < 3)) || ranks+=(3)
((np == 1 || np == 3)) || ranks+=("$np")
for c in empty sparse plain dense; do
    input="$out/inputs/$c.txt"
    golden="$out/results/$c.golden.bin"
    srun -N1 -n1 -c1 --hint=nomultithread --kill-on-bad-exit=1 --time="${STEP_TIME:-00:05:00}" \
        "$out/bin/reference" "$input" "$golden" </dev/null > "$out/logs/reference-$c.out" 2> "$out/logs/reference-$c.err"
    for d in "${divisors[@]}"; do
        for p in "${ranks[@]}"; do run_one custom "$c" normal "$p" 1 "$d" "$input" "$golden"; done
    done
done
if [[ "$mode" != bucket ]]; then
    i=0
    for knob in PP_FASTGEN=0 PP_RELABEL=0 PP_BATCH=1 PP_WITNESS_LIGHT=0 PP_WITNESS_TAU=255; do
        i=$((i + 1))
        run_one boundary dense normal "$np" "$i" 4 "$out/inputs/dense.txt" "$out/results/dense.golden.bin" "$knob"
    done
fi
if [[ "$mode" != sample ]]; then
    for binary in bucket atomic; do
        run_one custom dense "$binary" "$np" 1 1 "$out/inputs/dense.txt" "$out/results/dense.golden.bin"
    done
fi
((failed == 0)) || { echo 'Correctness gate failed; public runs cancelled.' >&2; exit 1; }

ci=0
for c in "${cases[@]}"; do
    cp "$case_dir/$c.txt" "$out/inputs/$c.txt"
    input="$out/inputs/$c.txt"
    golden="$case_dir/$c.golden.bin"
    if [[ "$mode" != bucket ]]; then
        for ((rep=1; rep<=repeat; rep++)); do
            # Rotate all three normal variants; each is first once in three repetitions.
            for ((j=0; j<${#divisors[@]}; j++)); do
                d="${divisors[$(((j + rep - 1 + ci) % ${#divisors[@]}))]}"
                run_one timing "$c" normal "$np" "$rep" "$d" "$input" "$golden" PP_WITNESS_REPORT=0
            done
        done
    fi
    for d in "${divisors[@]}"; do run_one profile "$c" phase "$np" 1 "$d" "$input" "$golden"; done
    if [[ "$mode" != sample ]]; then
        for binary in bucket atomic; do run_one profile "$c" "$binary" "$np" 1 1 "$input" "$golden"; done
    fi
    ((failed == 0)) || { echo 'Public run failed; stopping rather than accepting partial medians.' >&2; exit 1; }
    ci=$((ci + 1))
done
echo "== $passed passed, $failed failed =="
completed=1
