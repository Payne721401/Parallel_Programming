#!/usr/bin/env bash
set -euo pipefail

# HW 2-1 tests, for SEIREN. Run from anywhere: the script works in hw2/hw2-1. Run it inside one
# allocation, so each srun is a job step instead of a fresh job in the queue:
#
#   salloc -N1 -n24 -c1 --hint=nomultithread -t 60
#   bash ../tests/run_hw2-1.sh public     # judge shape, every public case against its golden
#   bash ../tests/run_hw2-1.sh custom     # edge cases against the starter, at several rank counts
#   bash ../tests/run_hw2-1.sh knobs      # the custom cases again under each PP_* knob
#   bash ../tests/run_hw2-1.sh all        # all three
#
# Env overrides:
#   CASES="01 04 10"   public subset (default: all 17)
#   REPEAT=3           runs per public case; the table shows the best and the median
#   RANKS="1 2 3 5 24" rank counts for custom
#   KNOBS="PP_BATCH=1 PP_BATCH=3"   one extra pass of the custom cases per knob
#   KNOB_RANKS="5 24"  rank counts for the knob passes
#   TIMING=1           pass PP_TIMING=1 and show the per-phase breakdown
#   CASES_DIR, OUT_ROOT
#
# Every run is under `ulimit -d 1228800`, the judge's 1200 MiB private-data cap. Exits nonzero
# if anything fails.

mode="${1:-public}"
case "$mode" in
    public|custom|knobs|all) ;;
    *)
        echo "usage: $0 [public|custom|knobs|all]" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
prob_dir="$(cd "$script_dir/../hw2-1" && pwd)"
orig_dir="$(cd "$script_dir/../original_version" && pwd)"
cd "$prob_dir"

cases_dir="${CASES_DIR:-cases}"
if [[ ! -d "$cases_dir/hw2-1" && -d /home/share/pp26/cases/hw2-1 ]]; then
    cases_dir=/home/share/pp26/cases
fi
out_root="${OUT_ROOT:-tmp/hw2-1-tests}"
repeat="${REPEAT:-1}"
public_cases="${CASES:-01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17}"
ranks="${RANKS:-1 2 3 5 24}"
knobs="${KNOBS:-PP_BATCH=1 PP_BATCH=3}"
knob_ranks="${KNOB_RANKS:-5 24}"

if [[ "${NO_ULIMIT:-0}" != "1" ]]; then
    ulimit -d 1228800   # NO_ULIMIT=1 only for dry runs on machines without RLIMIT_DATA
fi

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing command: $1" >&2
        exit 1
    fi
}
check_command srun
check_command hw2-1-check

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "missing required path: $1" >&2
        exit 1
    fi
}

judge_shape=(srun -N1 -n24 -c1 --hint=nomultithread)
passed=0
failed=0
failures=()

# run_one <label> <log prefix> <golden> <input> <output> <env...> -- <launcher...>
# Runs the program once, checks it, and prints one line. Leaves time/threads in $last_time and
# $last_threads.
run_one() {
    local label="$1" log="$2" golden="$3" input="$4" output="$5"
    shift 5
    local envs=()
    while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
    shift
    [[ "${TIMING:-0}" == "1" ]] && envs+=(PP_TIMING=1)

    local status
    rm -f "$output"
    # </dev/null: srun forwards stdin to task 0, which would swallow the case list being read.
    if env ${envs[@]+"${envs[@]}"} "$@" ./hw2-1 "$input" "$output" </dev/null >"$log.out" 2>"$log.err"; then
        if [[ ! -f "$output" ]]; then
            status=NO
        elif hw2-1-check "$golden" "$output" >"$log.check" 2>&1; then
            status=PASS
        else
            status=WA
        fi
    else
        status=RE
    fi
    last_time="$(sed -n 's/^Compute time: \([0-9.eE+-]*\) s$/\1/p' "$log.out")"
    last_threads="$(sed -n 's/^Threads: \([0-9-]*\).*/\1/p' "$log.out")"
    last_time="${last_time:--}"
    last_threads="${last_threads:--}"
    printf '%-4s %-34s time %-12s threads %s\n' "$status" "$label" "$last_time" "$last_threads"
    if [[ "${TIMING:-0}" == "1" ]]; then grep '^\[hw2-1\]' "$log.err" | sed 's/^/       /' || true; fi
    if [[ "$status" == PASS ]]; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
        failures+=("$label ($status, see $log.*)")
        sed 's/^/       /' "$log.err" | tail -n 5 || true
    fi
    last_status="$status"
}

build() {
    make clean >/dev/null
    make
}

build_seq() {
    make -C "$orig_dir" hw2-1-seq
}

run_public() {
    local dir="$out_root/public"
    mkdir -p "$dir"
    local summary="$dir/summary.tsv"
    [[ -f "$summary" ]] && mv "$summary" "$dir/summary.prev.tsv"
    printf 'case\trep\tstatus\ttime\tthreads\n' >"$summary"

    echo "== public cases, judge shape: ${judge_shape[*]}, REPEAT=$repeat =="
    local c rep
    for c in $public_cases; do
        local input="$cases_dir/hw2-1/$c.txt" golden="$cases_dir/hw2-1/$c.golden.bin"
        require_path "$input"
        require_path "$golden"
        for ((rep = 1; rep <= repeat; rep++)); do
            run_one "public $c #$rep" "$dir/$c.$rep" "$golden" "$input" "$dir/$c.out.bin" -- "${judge_shape[@]}"
            printf '%s\t%s\t%s\t%s\t%s\n' "$c" "$rep" "$last_status" "$last_time" "$last_threads" >>"$summary"
        done
    done

    # Best and median per case, next to the previous run's.
    echo "== public summary (seconds; prev = last run of this mode) =="
    printf '%-5s %-10s %-10s %-10s %s\n' case best median prev_best status
    for c in $public_cases; do
        local now prev st
        now="$(awk -F'\t' -v c="$c" '$1==c && $4!="-"{print $4}' "$summary" | sort -g)"
        st="$(awk -F'\t' -v c="$c" '$1==c{print $3}' "$summary" | sort -u | paste -sd, -)"
        local best="-" median="-"
        if [[ -n "$now" ]]; then
            best="$(head -n1 <<<"$now")"
            median="$(awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}' <<<"$now")"
        fi
        prev="-"
        if [[ -f "$dir/summary.prev.tsv" ]]; then
            prev="$(awk -F'\t' -v c="$c" '$1==c && $4!="-"{print $4}' "$dir/summary.prev.tsv" | sort -g | head -n1)"
            prev="${prev:--}"
        fi
        printf '%-5s %-10s %-10s %-10s %s\n' "$c" "$best" "$median" "$prev" "$st"
    done
}

# Edge cases of section 2.5: scale 1, repeated and isolated sources, nsample 0 and N, S = 32,
# both scrambles, edgefactor 256, and graphs sparse enough to be mostly leaves and islands.
custom_inputs() {
    cat <<'EOF'
s1_dup      1 1 0 0 2 3 0 1 0
s1_iso      1 1 5 1 1 2 1 1
s3_rep      3 1 7 1 8 4 5 5 5 2
s7_nosamp   7 1 3 0 0 1 5
s7_one      7 1 9 1 1 4 1 2 3 4
s8_32src    8 1 11 1 256 32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 255
s10_ef8     10 8 99 0 1024 5 0 1 2 1023 512
s12_sparse  12 1 77 1 4096 32 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 4095
s12_ef256   12 256 5 1 100 8 0 1 2 3 4000 4001 4002 4095
s14_full    14 16 42 0 16384 6 0 1 2 3 9999 16383
s16_spec    16 16 1 1 100 2 0 12345
s16_raw     16 16 2024 0 65536 4 0 1 2 3
s16_dup     16 16 7 1 65536 3 777 777 30000
s18_mix     18 16 31337 1 1000 8 0 1 2 3 65536 131072 200000 262143
s20_raw     20 16 5 0 1000 4 0 1 2 3
s20_ef4     20 4 9 1 1000 6 10 20 30 40 50 1048575
EOF
}

make_goldens() {
    local dir="$1" name args
    mkdir -p "$dir"
    while read -r name args; do
        echo "$args" >"$dir/$name.txt"
        if [[ ! -f "$dir/$name.golden.bin" || "$dir/$name.txt" -nt "$dir/$name.golden.bin" ]]; then
            srun -N1 -n1 "$orig_dir/hw2-1-seq" "$dir/$name.txt" "$dir/$name.golden.bin" </dev/null >/dev/null
        fi
    done < <(custom_inputs)
}

# run_custom <title> <rank counts> <env...>
run_custom() {
    local title="$1" rks="$2"
    shift 2
    local dir="$out_root/custom" n name args
    echo "== $title =="
    while read -r name args; do
        for n in $rks; do
            run_one "$name -n$n${1:+ $*}" "$dir/$name.n$n${1:+.${1//=/}}" "$dir/$name.golden.bin" \
                "$dir/$name.txt" "$dir/$name.out.bin" "$@" -- srun -N1 -n"$n" -c1 --hint=nomultithread
        done
    done < <(custom_inputs)
}

build
if [[ "$mode" == public || "$mode" == all ]]; then
    run_public
fi
if [[ "$mode" == custom || "$mode" == knobs || "$mode" == all ]]; then
    build_seq
    make_goldens "$out_root/custom"
fi
if [[ "$mode" == custom || "$mode" == all ]]; then
    run_custom "custom cases vs starter, ranks: $ranks" "$ranks"
fi
if [[ "$mode" == knobs || "$mode" == all ]]; then
    for k in $knobs; do
        run_custom "custom cases with $k, ranks: $knob_ranks" "$knob_ranks" "$k"
    done
fi

echo "== $passed passed, $failed failed =="
if ((failed)); then
    printf '  %s\n' "${failures[@]}"
    exit 1
fi
