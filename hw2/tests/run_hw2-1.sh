#!/usr/bin/env bash
set -euo pipefail

# HW 2-1 tests and measurements, for SEIREN. Run from anywhere: the script works in hw2/hw2-1.
# It must run inside one allocation, where each srun is a job step; outside one, every srun would
# be a fresh job in the queue, so the script refuses. Hand the script to salloc, so the
# allocation ends with it (and a refused salloc runs nothing):
#
#   salloc -N1 -n8 -c1 --hint=nomultithread --mem=32G -t 30 bash ../tests/run_hw2-1.sh <mode>
#
# Modes:
#   public    public cases against their goldens, at PUBLIC_NP ranks
#   custom    edge cases against the starter's output, at several rank counts
#   knobs     the custom cases again under each PP_* knob
#   all       public, custom and knobs
#   ab        two versions alternating on the same node in the same allocation, the only fair way
#             to tell whether a change made things faster: this hw2-1.cpp against an earlier one
#             from git, or (AB_KNOB) the same program with and without an environment knob
#   scaling   one case at 1, 2, 4, 8 ranks with the profiling build: which phases stop scaling
#   sysinfo   the compute node: CPU, caches, NUMA, rank binding, /dev/shm, huge pages, perf
#
# Two builds. ./hw2-1 is the judge's (no measurement compiled in); public, custom, knobs and ab
# time it. ./hw2-1-prof is built with -DHW21_PROF and prints per-phase times, page faults,
# window alloc/free times and the traversal's parts; TIMING=1, STATS=1 and scaling use it. Its
# Compute time includes the measurement, so compare times only between runs of the same build.
#
# The practice QoS `hw` allows at most 8 tasks (cpu=16 with nomultithread) and 32G per job, so 09
# and 10 are left to `hw2-1-judge`, which also gives the official 24-rank times (`hw2-1-judge 09`:
# one case, no cooldown, scoreboard untouched).
#
# Env overrides:
#   CASES="01 04"      cases (public default: all but 09 and 10, CASES=all for every case;
#                      ab default: 04 05 07 08 12 13)
#   PUBLIC_NP=8        ranks for the public and ab runs
#   REPEAT=3           runs per case (ab default 3); tables show the best and the median
#   AB_REF=<git rev>   ab's baseline (default: HEAD if hw2-1.cpp has uncommitted changes, else
#                      the version before its last commit)
#   AB_KNOB=PP_X=v     ab between this program with PP_X=v (base) and without it (new), no git
#   RANKS="1 3 8"      rank counts for custom
#   KNOBS="PP_BATCH=1 PP_BATCH=3 PP_FASTGEN=0 PP_RELABEL=0"   one extra pass of the custom cases per knob
#   KNOB_RANKS="8"     rank counts for the knob passes
#   TIMING=1           public runs use the profiling build and show its breakdown
#   STATS=1            ... plus the graph's shape (PP_STATS=1; slow, so times mean little)
#   SCALE_CASE=04      scaling's case; SCALE_NPS="1 2 4 8" its rank counts
#   VERBOSE=1          custom and knob runs print a line per run (default: a dot per pass)
#   CASES_DIR, OUT_ROOT
# Other PP_* knobs pass through from the environment; the profiling build's experiments are
# PP_PRETOUCH=1 (first-touch the CSR before the fill) and PP_PROBE=1 (time the fill's reads alone).
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
    public|custom|knobs|all|ab|scaling|sysinfo) ;;
    *)
        echo "usage: $0 [public|custom|knobs|all|ab|scaling|sysinfo]" >&2
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
knobs="${KNOBS:-PP_BATCH=1 PP_BATCH=3 PP_FASTGEN=0 PP_RELABEL=0}"
knob_ranks="${KNOB_RANKS:-8}"
scale_case="${SCALE_CASE:-04}"
scale_nps="${SCALE_NPS:-1 2 4 8}"

if [[ -z "${SLURM_JOB_ID:-}" && "${OUTSIDE_OK:-0}" != "1" ]]; then
    echo "not inside an allocation: every srun would queue as its own job. Run it as" >&2
    echo "  salloc -N1 -n8 -c1 --hint=nomultithread --mem=32G -t 30 bash $0 $mode" >&2
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

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "missing required path: $1" >&2
        exit 1
    fi
}

public_shape=(srun -N1 -n"$public_np" -c1 --hint=nomultithread)
bin=./hw2-1    # the program run_one runs
passed=0
failed=0
failures=()
timing_now=0   # the runs at hand use the profiling build: show its breakdown
quiet_now=0    # passes print a dot instead of a line (custom and knobs, unless VERBOSE=1)
dots_open=0    # a line of dots is waiting for its newline

# The profiling build's report, condensed: the graph's size, phases (slowest rank), page faults
# (most on a rank), the traversal's parts, window alloc/free of 5 ms or more, counters, and any
# stats lines.
compact_timing() {
    awk '
        /^\[hw2-1\] P=/ { print "graph: " substr($0, 9) }
        /^\[hw2-1\] [a-z][a-z:-]* +max/ {
            name = $2; t = $4
            if (name ~ /^trav:/)      trav = trav sprintf("  %s %s", substr(name, 6), t)
            else if (name ~ /:/)      { if (t + 0 >= 0.005) win = win sprintf("  %s %s", name, t) }
            else {
                ph = ph sprintf("  %s %s", name, t)
                if ($NF + 0 > 0) flt = flt sprintf("  %s %s", name, $NF)
            }
        }
        /^\[hw2-1\] [a-z-]+ +sum/ { cnt = cnt sprintf("  %s %s", $2, $4) }
        /^\[hw2-1\] stats:/       { st = st "\n" substr($0, 9) }
        END {
            print "phases (s):" ph
            print "faults:" flt
            print "traverse (s):" trav
            print "windows (s):" win
            printf "counts:%s%s\n", cnt, st
        }' "$1"
}

# run_one <label> <log prefix> <golden> <input> <output> <env...> -- <launcher...>
# Runs $bin once and checks its output with hw2-1-check against the golden. Leaves time/threads
# in $last_time and $last_threads.
run_one() {
    local label="$1" log="$2" golden="$3" input="$4" output="$5"
    shift 5
    local envs=()
    while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
    shift

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
    if ((timing_now)); then compact_timing "$log.err" | sed 's/^/       /'; fi
    if [[ "$status" != PASS ]]; then
        failed=$((failed + 1))
        failures+=("$label ($status, see $log.*)")
        sed 's/^/       /' "$log.err" | tail -n 5 || true
    fi
}

build() {
    make
}

# The profiling build, from the same objects plus hw2-1.cpp compiled with -DHW21_PROF.
build_prof() {
    if [[ ! -x hw2-1-prof || hw2-1.cpp -nt hw2-1-prof || main.o -nt hw2-1-prof ]]; then
        mpicxx -std=c++17 -O3 -march=native -DHW21_PROF -c hw2-1.cpp -o hw2-1-prof.o
        mpicxx -std=c++17 -O3 -march=native -o hw2-1-prof main.o graph_gen.o hw2-1-prof.o
    fi
}

build_seq() {
    make -C "$orig_dir" hw2-1-seq
}

# median of the numbers on stdin, one per line
median() {
    sort -g | awk '{a[NR]=$1} END{if (NR) print a[int((NR+1)/2)]; else print "-"}'
}

run_public() {
    local dir="$out_root/public"
    mkdir -p "$dir"
    local summary="$dir/summary.tsv" envs=()
    [[ -f "$summary" ]] && mv "$summary" "$dir/summary.prev.tsv"
    printf 'case\trep\tstatus\ttime\tthreads\n' >"$summary"

    if [[ "${TIMING:-0}" == "1" || "${STATS:-0}" == "1" ]]; then
        build_prof
        bin=./hw2-1-prof
        timing_now=1
        [[ "${STATS:-0}" == "1" ]] && envs+=(PP_STATS=1)
    fi
    echo "== public cases: $bin, ${public_shape[*]}, REPEAT=$repeat =="
    quiet_now=0
    local c rep
    for c in $public_cases; do
        local input="$cases_dir/hw2-1/$c.txt" golden="$cases_dir/hw2-1/$c.golden.bin"
        require_path "$input"
        require_path "$golden"
        for ((rep = 1; rep <= repeat; rep++)); do
            run_one "public $c #$rep" "$dir/$c.$rep" "$golden" "$input" "$dir/$c.out.bin" \
                ${envs[@]+"${envs[@]}"} -- "${public_shape[@]}"
            printf '%s\t%s\t%s\t%s\t%s\n' "$c" "$rep" "$last_status" "$last_time" "$last_threads" >>"$summary"
        done
    done
    timing_now=0
    bin=./hw2-1

    # Best and median per case, next to the previous run's.
    echo "== public summary (seconds; prev = last run of this mode, whichever build it used) =="
    printf '%-5s %-10s %-10s %-10s %s\n' case best median prev_best status
    for c in $public_cases; do
        local now prev st best="-" med="-"
        now="$(awk -F'\t' -v c="$c" '$1==c && $4!="-"{print $4}' "$summary")"
        st="$(awk -F'\t' -v c="$c" '$1==c{print $3}' "$summary" | sort -u | paste -sd, -)"
        if [[ -n "$now" ]]; then
            best="$(sort -g <<<"$now" | head -n1)"
            med="$(median <<<"$now")"
        fi
        prev="-"
        if [[ -f "$dir/summary.prev.tsv" ]]; then
            prev="$(awk -F'\t' -v c="$c" '$1==c && $4!="-"{print $4}' "$dir/summary.prev.tsv" | sort -g | head -n1)"
            prev="${prev:--}"
        fi
        printf '%-5s %-10s %-10s %-10s %s\n' "$c" "$best" "$med" "$prev" "$st"
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
    local dir="$out_root/ab" base_bin base_env=()
    mkdir -p "$dir"
    if [[ -n "${AB_KNOB:-}" ]]; then
        base_bin=./hw2-1
        base_env=("$AB_KNOB")
        echo "== ab: base = ./hw2-1 with $AB_KNOB"
        echo "         new  = ./hw2-1 without it; ${public_shape[*]}, REPEAT=$repeat =="
    else
        local ref
        ref="$(ab_ref)"
        git show "$ref:./hw2-1.cpp" >"$dir/base.cpp"
        mpicxx -std=c++17 -O3 -march=native -I. -c "$dir/base.cpp" -o "$dir/base.o"
        mpicxx -std=c++17 -O3 -march=native -o "$dir/hw2-1-base" main.o graph_gen.o "$dir/base.o"
        base_bin="$dir/hw2-1-base"
        echo "== ab: base = hw2-1.cpp at $(git log -n 1 --format='%h "%s"' "$ref" | cut -c1-70)"
        echo "         new  = hw2-1.cpp in the working tree; ${public_shape[*]}, REPEAT=$repeat =="
    fi

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
                if [[ "$which" == base ]]; then
                    bin="$base_bin"
                    run_one "ab $c base #$rep" "$dir/$c.base.$rep" "$golden" "$input" "$dir/$c.out.bin" \
                        ${base_env[@]+"${base_env[@]}"} -- "${public_shape[@]}"
                else
                    bin=./hw2-1
                    run_one "ab $c new #$rep" "$dir/$c.new.$rep" "$golden" "$input" "$dir/$c.out.bin" \
                        -- "${public_shape[@]}"
                fi
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

# One case at several rank counts with the profiling build; the table shows each phase's slowest
# rank per rank count and the speedup from the first to the last (ideal: the ratio of ranks).
run_scaling() {
    build_prof
    local c="$scale_case" dir="$out_root/scaling" np logs=() first last
    local input="$cases_dir/hw2-1/$c.txt" golden="$cases_dir/hw2-1/$c.golden.bin"
    require_path "$input"
    require_path "$golden"
    mkdir -p "$dir"
    first="${scale_nps%% *}"
    last="${scale_nps##* }"
    echo "== scaling: case $c at ranks $scale_nps, profiling build =="
    bin=./hw2-1-prof
    timing_now=1
    for np in $scale_nps; do
        run_one "case $c -n$np" "$dir/$c.n$np" "$golden" "$input" "$dir/$c.out.bin" \
            -- srun -N1 -n"$np" -c1 --hint=nomultithread
        logs+=("$dir/$c.n$np.err")
    done
    timing_now=0
    bin=./hw2-1
    echo "== scaling table: slowest rank's seconds (rows under 1 ms dropped); speedup P=$first -> P=$last (ideal $((last / first))x) =="
    awk -v nps="$scale_nps" '
        FNR == 1 { k++ }
        /^\[hw2-1\] [a-z][a-z:-]* +max/ {
            if (!($2 in seen)) { seen[$2] = ++n; order[n] = $2 }
            t[$2, k] = $4
        }
        END {
            split(nps, P, " ")
            printf "%-14s", "phase"
            for (i = 1; i <= k; i++) printf " %9s", "P=" P[i]
            printf " %8s\n", "speedup"
            for (j = 1; j <= n; j++) {
                nm = order[j]
                big = 0
                for (i = 1; i <= k; i++) if (((nm, i) in t) && t[nm, i] + 0 >= 0.001) big = 1
                if (!big) continue   # under 1 ms at every rank count
                printf "%-14s", nm
                for (i = 1; i <= k; i++) printf " %9s", ((nm, i) in t) ? t[nm, i] : "-"
                s = ((nm, 1) in t && (nm, k) in t && t[nm, k] + 0 > 0) ? sprintf("%.2fx", t[nm, 1] / t[nm, k]) : "-"
                printf " %8s\n", s
            }
        }' "${logs[@]}"
}

# The compute node, as the ranks see it.
run_sysinfo() {
    echo "== CPU and caches =="
    srun -N1 -n1 lscpu </dev/null | grep -E "Model name|^Socket|Core\(s\)|Thread\(s\)|NUMA node|L1d|L2|L3|MHz" || true
    echo "== NUMA =="
    srun -N1 -n1 bash -c 'command -v numactl >/dev/null && numactl -H || echo "numactl not installed"' </dev/null
    echo "== binding of $public_np ranks (srun -c1 --hint=nomultithread) =="
    srun -N1 -n"$public_np" -c1 --hint=nomultithread \
        bash -c 'echo "rank $SLURM_PROCID cpus $(grep Cpus_allowed_list /proc/self/status | cut -f2)"' </dev/null | sort -n -k2
    echo "== shared memory and huge pages =="
    srun -N1 -n1 bash -c 'df -h /dev/shm; free -g; for f in enabled shmem_enabled; do
        echo "transparent_hugepage/$f: $(cat /sys/kernel/mm/transparent_hugepage/$f 2>/dev/null)"; done' </dev/null
    echo "== Open MPI shared-memory settings =="
    ompi_info --param shmem all --level 9 2>/dev/null | grep -iE "priority|backing|relocate" | head -n 8 || true
    ompi_info --param osc sm --level 9 2>/dev/null | grep -iE "backing|dir" | head -n 4 || true
    echo "== perf =="
    srun -N1 -n1 bash -c 'command -v perf || echo "perf not installed";
        echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"' </dev/null
}

if [[ "$mode" == sysinfo ]]; then
    run_sysinfo
    exit 0
fi

check_command hw2-1-check
build
case "$mode" in
    ab) run_ab ;;
    scaling) run_scaling ;;
esac
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
