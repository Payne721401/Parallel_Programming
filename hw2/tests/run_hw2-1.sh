#!/usr/bin/env bash
set -euo pipefail

# HW 2-1 tests, for SEIREN. Run from anywhere: the script works in hw2/hw2-1. It must run inside
# one allocation, where each srun is a job step; outside one, every srun would be a fresh job in
# the queue, so the script refuses. Hand the script to salloc, so the allocation ends with it
# (and a refused salloc runs nothing):
#
#   salloc -N1 -n8 -c1 --hint=nomultithread -t 30 bash ../tests/run_hw2-1.sh all
#
# Modes: public (public cases against their goldens, at PUBLIC_NP ranks), custom (edge cases
# against the starter, at several rank counts), knobs (the custom cases under each PP_* knob),
# all (the three), and ab: this hw2-1.cpp against an earlier one from git, the two alternating
# on the same node in the same allocation, which is the only fair way to tell whether a change
# made things faster. The practice QoS `hw` allows at most 8 tasks (cpu=16 with nomultithread)
# and 32G per job, so 09 and 10 are left to `hw2-1-judge`, which also gives the official
# 24-rank times (`hw2-1-judge 09`: one case, no cooldown, scoreboard untouched).
#
#   salloc -N1 -n8 -c1 --hint=nomultithread --mem=32G -t 30 bash ../tests/run_hw2-1.sh ab
#
# Env overrides:
#   CASES="01 04"      cases (public default: all but 09 and 10, CASES=all for every case;
#                      ab default: 04 05 07 08 12 13)
#   PUBLIC_NP=8        ranks for the public and ab runs
#   REPEAT=3           runs per case (ab default 3); tables show the best and the median
#   AB_REF=<git rev>   ab's baseline (default: HEAD if hw2-1.cpp has uncommitted changes, else
#                      the version before its last commit)
#   RANKS="1 3 8"      rank counts for custom
#   KNOBS="PP_BATCH=1 PP_BATCH=3"   one extra pass of the custom cases per knob
#   KNOB_RANKS="8"     rank counts for the knob passes
#   TIMING=1           public runs get PP_TIMING=1; each shows its phases' slowest rank on one line
#   VERBOSE=1          custom and knob runs print a line per run (default: a dot per pass)
#   CASES_DIR, OUT_ROOT
#
# Every run's output is checked with hw2-1-check, which compares every field (n_reached,
# checksum, both digests, every sampled distance), against the course's golden for the public
# cases and against the starter's own output (hw2-1-seq, one rank) for the custom ones. Logs of
# each run stay under OUT_ROOT.
#
# Every run is under `ulimit -d 1228800`, the judge's 1200 MiB private-data cap. Exits nonzero
# if anything fails.

mode="${1:-public}"
case "$mode" in
    public|custom|knobs|all|ab) ;;
    *)
        echo "usage: $0 [public|custom|knobs|all|ab]" >&2
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
if [[ "$mode" == ab ]]; then
    repeat="${REPEAT:-3}"
    public_cases="${CASES:-04 05 07 08 12 13}"
else
    repeat="${REPEAT:-1}"
    public_cases="${CASES:-01 02 03 04 05 06 07 08 11 12 13 14 15 16 17}"
fi
if [[ "$public_cases" == all ]]; then
    public_cases="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17"
fi
public_np="${PUBLIC_NP:-8}"
ranks="${RANKS:-1 3 8}"
knobs="${KNOBS:-PP_BATCH=1 PP_BATCH=3}"
knob_ranks="${KNOB_RANKS:-8}"

if [[ -z "${SLURM_JOB_ID:-}" && "${OUTSIDE_OK:-0}" != "1" ]]; then
    echo "not inside an allocation: every srun would queue as its own job. Run it as" >&2
    echo "  salloc -N1 -n8 -c1 --hint=nomultithread -t 30 bash $0 $mode" >&2
    exit 1
fi

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

public_shape=(srun -N1 -n"$public_np" -c1 --hint=nomultithread)
bin=./hw2-1    # the program run_one runs (ab swaps in the baseline)
passed=0
failed=0
failures=()
timing_now=0   # PP_TIMING for this batch of runs (the public ones, when TIMING=1)
quiet_now=0    # passes print a dot instead of a line (custom and knobs, unless VERBOSE=1)
dots_open=0    # a line of dots is waiting for its newline

# The PP_TIMING breakdown on one line: each phase's slowest rank, then the round count.
compact_timing() {
    awk '/^\[hw2-1\] [a-z]+ +max/ { printf "%s %s  ", $2, $4 }
         /^\[hw2-1\] rounds/      { printf "rounds %s", $4 }
         END                      { print "" }' "$1"
}

# run_one <label> <log prefix> <golden> <input> <output> <env...> -- <launcher...>
# Runs the program once and checks its output with hw2-1-check against the golden. Leaves
# time/threads in $last_time and $last_threads.
run_one() {
    local label="$1" log="$2" golden="$3" input="$4" output="$5"
    shift 5
    local envs=()
    while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
    shift
    ((timing_now)) && envs+=(PP_TIMING=1)

    local status
    rm -f "$output"
    # </dev/null: srun forwards stdin to task 0, which would swallow the case list being read.
    if env ${envs[@]+"${envs[@]}"} "$@" "$bin" "$input" "$output" </dev/null >"$log.out" 2>"$log.err"; then
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
    last_status="$status"
    if [[ "$status" == PASS ]]; then
        passed=$((passed + 1))
        if ((quiet_now)) && [[ "${VERBOSE:-0}" != "1" ]]; then
            printf '.'
            dots_open=1
            return
        fi
    fi
    ((dots_open)) && echo
    dots_open=0
    printf '%-4s %-34s time %-12s threads %s\n' "$status" "$label" "$last_time" "$last_threads"
    if ((timing_now)); then printf '       %s\n' "$(compact_timing "$log.err")"; fi
    if [[ "$status" != PASS ]]; then
        failed=$((failed + 1))
        failures+=("$label ($status, see $log.*)")
        sed 's/^/       /' "$log.err" | tail -n 5 || true
    fi
}

build() {
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

    echo "== public cases: ${public_shape[*]}, REPEAT=$repeat =="
    [[ "${TIMING:-0}" == "1" ]] && timing_now=1
    quiet_now=0
    local c rep
    for c in $public_cases; do
        local input="$cases_dir/hw2-1/$c.txt" golden="$cases_dir/hw2-1/$c.golden.bin"
        require_path "$input"
        require_path "$golden"
        for ((rep = 1; rep <= repeat; rep++)); do
            run_one "public $c #$rep" "$dir/$c.$rep" "$golden" "$input" "$dir/$c.out.bin" -- "${public_shape[@]}"
            printf '%s\t%s\t%s\t%s\t%s\n' "$c" "$rep" "$last_status" "$last_time" "$last_threads" >>"$summary"
        done
    done
    timing_now=0

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
        if [[ ! -f "$dir/$name.txt" || "$(cat "$dir/$name.txt")" != "$args" ]]; then
            echo "$args" >"$dir/$name.txt"
        fi
        if [[ ! -f "$dir/$name.golden.bin" || ! -f "$dir/$name.seq.out" || "$dir/$name.txt" -nt "$dir/$name.golden.bin" ]]; then
            srun -N1 -n1 "$orig_dir/hw2-1-seq" "$dir/$name.txt" "$dir/$name.golden.bin" </dev/null >"$dir/$name.seq.out"
        fi
    done < <(custom_inputs)
    # The judge reads the thread count main prints; ours must not exceed what MPI alone starts.
    echo "starter threads: $(sed -n 's/^Threads: \([0-9-]*\).*/\1/p' "$dir"/*.seq.out | sort -u | paste -sd, -)"
}

# run_custom <title> <rank counts> <env...>
run_custom() {
    local title="$1" rks="$2"
    shift 2
    local dir="$out_root/custom" n name args before=$failed runs=0
    echo "== $title =="
    quiet_now=1
    while read -r name args; do
        for n in $rks; do
            run_one "$name -n$n${1:+ $*}" "$dir/$name.n$n${1:+.${1//=/}}" "$dir/$name.golden.bin" \
                "$dir/$name.txt" "$dir/$name.out.bin" "$@" -- srun -N1 -n"$n" -c1 --hint=nomultithread
            runs=$((runs + 1))
        done
    done < <(custom_inputs)
    quiet_now=0
    ((dots_open)) && echo
    dots_open=0
    echo "   $runs runs, $((failed - before)) failed"
}

# median of the numbers on stdin, one per line
median() {
    sort -g | awk '{a[NR]=$1} END{if (NR) print a[int((NR+1)/2)]; else print "-"}'
}

# The baseline for ab: AB_REF, or HEAD when hw2-1.cpp has uncommitted changes, or else the
# version just before the last commit that changed it.
ab_ref() {
    if [[ -n "${AB_REF:-}" ]]; then
        echo "$AB_REF"
    elif ! git diff --quiet HEAD -- hw2-1.cpp; then
        echo HEAD
    else
        echo "$(git log -n 1 --format=%H -- hw2-1.cpp)^"
    fi
}

run_ab() {
    local dir="$out_root/ab" ref
    mkdir -p "$dir"
    ref="$(ab_ref)"
    git show "$ref:./hw2-1.cpp" >"$dir/base.cpp"
    mpicxx -std=c++17 -O3 -march=native -I. -c "$dir/base.cpp" -o "$dir/base.o"
    mpicxx -std=c++17 -O3 -march=native -o "$dir/hw2-1-base" main.o graph_gen.o "$dir/base.o"
    echo "== ab: base = hw2-1.cpp at $(git log -n 1 --format='%h "%s"' "$ref" | cut -c1-70)"
    echo "         new  = hw2-1.cpp in the working tree; ${public_shape[*]}, REPEAT=$repeat =="

    local tsv="$dir/ab.tsv" c rep which
    printf 'case\twhich\trep\tstatus\ttime\n' >"$tsv"
    quiet_now=1
    for c in $public_cases; do
        local input="$cases_dir/hw2-1/$c.txt" golden="$cases_dir/hw2-1/$c.golden.bin"
        require_path "$input"
        require_path "$golden"
        for ((rep = 1; rep <= repeat; rep++)); do
            # Alternate who goes first, so neither side always runs on a warmer node.
            local order="base new"
            ((rep % 2 == 0)) && order="new base"
            for which in $order; do
                if [[ "$which" == base ]]; then bin="$dir/hw2-1-base"; else bin=./hw2-1; fi
                run_one "ab $c $which #$rep" "$dir/$c.$which.$rep" "$golden" "$input" "$dir/$c.out.bin" \
                    -- "${public_shape[@]}"
                printf '%s\t%s\t%s\t%s\t%s\n' "$c" "$which" "$rep" "$last_status" "$last_time" >>"$tsv"
            done
        done
    done
    bin=./hw2-1
    quiet_now=0
    ((dots_open)) && echo
    dots_open=0

    echo "== ab summary (seconds; change = new median vs base median) =="
    printf '%-5s %-10s %-10s %-10s %-10s %s\n' case base_best base_med new_best new_med change
    for c in $public_cases; do
        local b n bb bm nb nm
        b="$(awk -F'\t' -v c="$c" '$1==c && $2=="base" && $5!="-"{print $5}' "$tsv")"
        n="$(awk -F'\t' -v c="$c" '$1==c && $2=="new" && $5!="-"{print $5}' "$tsv")"
        bb="$(sort -g <<<"$b" | head -n1)"; bm="$(median <<<"$b")"
        nb="$(sort -g <<<"$n" | head -n1)"; nm="$(median <<<"$n")"
        printf '%-5s %-10s %-10s %-10s %-10s %s\n' "$c" "${bb:--}" "$bm" "${nb:--}" "$nm" \
            "$(awk -v b="$bm" -v n="$nm" 'BEGIN{ if (b+0 > 0 && n+0 > 0) printf "%+.1f%%", (n/b - 1) * 100; else print "-" }')"
    done
}

build
if [[ "$mode" == ab ]]; then
    run_ab
fi
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
