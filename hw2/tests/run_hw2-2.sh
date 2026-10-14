#!/usr/bin/env bash
set -euo pipefail

# HW 2-2 tests and measurements, for SEIREN. Run from anywhere: the script works in hw2/hw2-2.
# It must run inside one allocation (every srun a job step): hand it to salloc or sbatch
# (tests/batch_hw2-2.sh runs it in the background). The practice QoS `hw` allows 16 CPUs (8
# ranks with nomultithread), 32G and 2 nodes, so two shapes:
#
#   salloc -N2 -n8 --ntasks-per-node=4 -c1 --hint=nomultithread --mem=16G -t 40 bash ../tests/run_hw2-2.sh <mode>
#   salloc -N1 -n8 -c1 --hint=nomultithread --mem=32G -t 20 bash ../tests/run_hw2-2.sh bench-mem
#
# (--mem is per node.) Modes:
#   public    every public case at the practice shape, -N2 -n8 with 4 ranks a node, checked
#             with hw2-2-check; TIMING=1 runs the profiling build (hw2-2-prof, -DHW22_PROF) and
#             shows its report: phases (slowest and fastest rank), products per rank, bytes sent
#   ab        normal builds, alternating AB_REF (default 1604d20) and current code;
#             CASES defaults to p01 p02 p03, REPEAT=3, PP_HOT=1 enables the candidate path
#   ranks     the correctness cases at uneven rank counts, on one node (-n1, -n3, -n4) and on
#             two (-n2, -n5, -n8), plus the custom cases once `custom` has made them
#   custom    builds tests/gen_spgemm, writes the boundaries no public case reaches (C complete
#             and products > 2^32; nnz(A) and K at their maxima; small random shapes) under
#             OUT_ROOT/custom, their C.csr from the starter at one rank, and checks ours on them
#   sysinfo   both nodes: CPU, NUMA, network devices and their state, Open MPI's transports
#   bench-mem tests/mpi_bench mem, 8 ranks on one node (the second shape): first touch and
#             release of private vs shared memory with 1, 2, 4, 8 ranks at once; copy bandwidth
#   bench-net tests/mpi_bench net, 4 ranks on each of two nodes: latency and bandwidth within a
#             node and across nodes, all pairs streaming across at once, MPI_Alltoall
#   stats     tests/csr_stats on CASES (default p01 p02 p03): sizes, work per row, rank and node
#             under row-wise and K-split distributions, cross-node traffic, partial-C sizes,
#             local kernel speed. Serial and uncapped (it holds a whole B)
#   baseline  the starter (hw2-2-seq) on CASES (default every public case) in the practice shape
#
# Every run of hw2-2 or the starter is under `ulimit -d 1228800`, the judge's 1200 MiB cap on
# private data. Exits nonzero if any run fails.
#
# Env overrides: CASES="c01 p01" (public, ranks, baseline, stats), CASES_DIR, OUT_ROOT,
# STATS_P="8 24" (rank counts for stats), SHAPE="-N2 -n8 --ntasks-per-node=4" (public),
# TIMING=1 (public), AB_REF and REPEAT (ab), CUSTOM_CASES="hot r1" (custom),
# VERBOSE=1 (ranks and custom print a line per run, not a dot). PP_* variables pass through.

mode="${1:-}"
case "$mode" in
    public|ab|ranks|custom|sysinfo|bench-mem|bench-net|stats|baseline) ;;
    *)
        echo "usage: $0 public|ab|ranks|custom|sysinfo|bench-mem|bench-net|stats|baseline" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
prob_dir="$(cd "$script_dir/../hw2-2" && pwd)"
orig_dir="$(cd "$script_dir/../original_version" && pwd)"
cd "$prob_dir"

cases_dir="${CASES_DIR:-cases}"
if [[ ! -d "$cases_dir/hw2-2" && -d /home/share/pp26/cases/hw2-2 ]]; then
    cases_dir=/home/share/pp26/cases
fi
out_root="${OUT_ROOT:-tmp/hw2-2-tests}"
bin_dir="$out_root/bin"
mkdir -p "$bin_dir"
FLAGS=(-std=c++17 -O3 -march=native)
correctness_cases="c01 c02 c03 c04 c05 c06 c07 c08 c09 c10 c11 c12 c13 c14"
public_cases="$correctness_cases p01 p02 p03"

if [[ -z "${SLURM_JOB_ID:-}" && "${OUTSIDE_OK:-0}" != "1" ]]; then
    echo "not inside an allocation: every srun would queue as its own job. Run it as" >&2
    echo "  salloc -N2 -n8 --ntasks-per-node=4 -c1 --hint=nomultithread --mem=16G -t 40 bash $0 $mode" >&2
    echo "  (bench-mem: salloc -N1 -n8 -c1 --hint=nomultithread --mem=32G -t 20 ...)" >&2
    exit 1
fi

# A case's directory: cases/hw2-2/<name>, or the one whose name starts with <name>_ (p01_large).
case_dir() {
    local d="$cases_dir/hw2-2/$1"
    if [[ ! -d "$d" ]]; then
        d="$(ls -d "$cases_dir/hw2-2/$1"_* 2>/dev/null | head -n1 || true)"
    fi
    if [[ -z "$d" || ! -f "$d/A.csr" ]]; then
        echo "missing case: $1 (under $cases_dir/hw2-2)" >&2
        exit 1
    fi
    echo "$d"
}

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing command: $1" >&2
        exit 1
    fi
}

passed=0
failed=0
failures=()
quiet_now=0    # passes print a dot instead of a line
dots_open=0    # a line of dots is waiting for its newline
last_status= last_time=

# The profiling build's report, condensed: phases, then counters.
compact_prof() {
    awk '
        /^\[hw2-2\] [a-z-]+ +max/ { ph = ph sprintf("  %s %s", $2, $4) }
        /^\[hw2-2\] [a-z-]+ +sum/ { cnt = cnt sprintf("  %s %s (max/mean %s)", $2, $4, $NF) }
        END { print "phases (s):" ph; print "counts:" cnt }' "$1"
}

# run_case <label> <case dir> <binary> <srun args...>: runs the binary on the case's A and B,
# checks its output against the case's C.csr, and prints a line (or a dot when quiet and fine).
run_case() {
    local label="$1" dir="$2" bin="$3"
    shift 3
    local log="$out_root/logs/${label// /_}" out="$out_root/out.csr" status t rss
    mkdir -p "$out_root/logs"
    rm -f "$out"
    # </dev/null: srun forwards stdin to task 0, which would swallow a loop's input.
    if (ulimit -d 1228800; srun "$@" -c1 --hint=nomultithread "$bin" "$dir/A.csr" "$dir/B.csr" "$out" \
            </dev/null >"$log.out" 2>"$log.err"); then
        if [[ ! -f "$out" ]]; then
            status=NO
        elif hw2-2-check "$dir/C.csr" "$out" >"$log.check" 2>&1; then
            status=PASS
        else
            status=WA
        fi
    else
        status=RE
    fi
    rm -f "$out"
    t="$(sed -n 's/^Compute time: \([0-9.]*\) s$/\1/p' "$log.out")"
    rss="$(sed -n 's/^Peak RSS: \([0-9]*\) KiB$/\1/p' "$log.out")"
    last_status="$status"
    last_time="${t:--}"
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
    printf '%-4s %-34s time %-12s peak RSS %s\n' "$status" "$label" "${t:--}" "${rss:+$((rss / 1024)) MiB}"
    if [[ "$bin" == *prof && "$status" == PASS ]]; then compact_prof "$log.err" | sed 's/^/       /'; fi
    if [[ "$status" != PASS ]]; then
        failed=$((failed + 1))
        failures+=("$label ($status, see $log.*)")
        { cat "$log.check" 2>/dev/null; tail -n 5 "$log.err"; } | sed 's/^/       /' || true
    fi
}

build() {
    make
}

# The profiling build: hw2-2.cpp with -DHW22_PROF.
build_prof() {
    if [[ ! -x hw2-2-prof || hw2-2.cpp -nt hw2-2-prof ]]; then
        mpicxx "${FLAGS[@]}" -DHW22_PROF -o hw2-2-prof main.cpp hw2-2.cpp
    fi
}

finish() {
    ((dots_open)) && echo
    echo "== $passed passed, $failed failed =="
    if ((failed)); then
        printf '  %s\n' "${failures[@]}"
        exit 1
    fi
}

run_public() {
    check_command hw2-2-check
    build
    local bin=./hw2-2 c
    if [[ "${TIMING:-0}" == "1" ]]; then
        build_prof
        bin=./hw2-2-prof
    fi
    read -r -a shape <<<"${SHAPE:--N2 -n8 --ntasks-per-node=4}"
    echo "== public cases: $bin, srun ${shape[*]} =="
    for c in ${CASES:-$public_cases}; do
        run_case "public $c" "$(case_dir "$c")" "$bin" "${shape[@]}"
    done
}

run_ab() {
    check_command hw2-2-check
    build
    local ref="${AB_REF:-1604d20}" repeat="${REPEAT:-3}" c rep which
    [[ "$repeat" =~ ^[1-9][0-9]*$ ]] || { echo "REPEAT must be positive" >&2; exit 2; }
    git show "$ref:hw2/hw2-2/hw2-2.cpp" >"$bin_dir/base.cpp"
    mpicxx "${FLAGS[@]}" -I. main.cpp "$bin_dir/base.cpp" -o "$bin_dir/base"
    read -r -a shape <<<"${SHAPE:--N2 -n8 --ntasks-per-node=4}"
    local tsv="$out_root/ab.tsv" dir bin
    printf 'case\twhich\trep\tstatus\ttime\n' >"$tsv"
    echo "== ab: baseline $ref, candidate working tree; srun ${shape[*]}, REPEAT=$repeat =="
    for c in ${CASES:-p01 p02 p03}; do
        dir="$(case_dir "$c")"
        for ((rep = 1; rep <= repeat; rep++)); do
            local order="base new"
            ((rep % 2 == 0)) && order="new base"
            for which in $order; do
                bin=./hw2-2
                [[ "$which" == base ]] && bin="$bin_dir/base"
                run_case "ab $c $which $rep" "$dir" "$bin" "${shape[@]}"
                printf '%s\t%s\t%s\t%s\t%s\n' "$c" "$which" "$rep" "$last_status" "$last_time" >>"$tsv"
            done
        done
    done
    local b n bb nb bm nm
    echo "== ab summary (seconds; change = new median vs base median) =="
    printf '%-5s %-11s %-11s %-11s %-11s %s\n' case base_best base_med new_best new_med change
    for c in ${CASES:-p01 p02 p03}; do
        b="$(awk -F'\t' -v c="$c" '$1==c && $2=="base" && $4=="PASS" && $5!="-" {print $5}' "$tsv" | sort -g)"
        n="$(awk -F'\t' -v c="$c" '$1==c && $2=="new" && $4=="PASS" && $5!="-" {print $5}' "$tsv" | sort -g)"
        bb="$(head -n1 <<<"$b")"; nb="$(head -n1 <<<"$n")"
        bm="$(awk 'NF {a[++n]=$1} END {if(n) print (a[int((n+1)/2)]+a[int((n+2)/2)])/2; else print "-"}' <<<"$b")"
        nm="$(awk 'NF {a[++n]=$1} END {if(n) print (a[int((n+1)/2)]+a[int((n+2)/2)])/2; else print "-"}' <<<"$n")"
        printf '%-5s %-11s %-11s %-11s %-11s %s\n' "$c" "${bb:--}" "$bm" "${nb:--}" "$nm" \
            "$(awk -v b="$bm" -v n="$nm" 'BEGIN {if(b+0>0 && n+0>0) printf "%+.1f%%",100*(n/b-1); else print "-"}')"
    done
    echo "Raw times: $tsv"
}

# The custom cases that exist so far: OUT_ROOT/custom/<name>/{A,B,C}.csr.
custom_dirs() {
    local d
    for d in "$out_root"/custom/*/; do
        [[ -f "$d/C.csr" ]] && echo "${d%/}"
    done
    return 0
}

run_ranks() {
    check_command hw2-2-check
    build
    local shapes=("-N1 -n1" "-N1 -n3" "-N1 -n4" "-N2 -n2 --ntasks-per-node=1" "-N2 -n5" "-N2 -n8 --ntasks-per-node=4")
    local s c d
    quiet_now=1
    for s in "${shapes[@]}"; do
        read -r -a shape <<<"$s"
        echo "== ranks: srun $s =="
        for c in ${CASES:-$correctness_cases}; do
            run_case "$c $s" "$(case_dir "$c")" ./hw2-2 "${shape[@]}"
        done
        while read -r d; do
            [[ -n "$d" ]] && run_case "$(basename "$d") $s" "$d" ./hw2-2 "${shape[@]}"
        done < <(custom_dirs)
        ((dots_open)) && echo
        dots_open=0
    done
    quiet_now=0
}

run_custom() {
    check_command hw2-2-check
    build
    make -C "$orig_dir" hw2-2-seq
    mpicxx "${FLAGS[@]}" -o "$bin_dir/gen_spgemm" "$script_dir/gen_spgemm.cpp"
    # name and generator arguments; the random shapes include M, K or N below the rank count
    local specs=(
        "hot hot"
        "dense dense"
        "nnza nnza"
        "r1 random 1 3 5 7"
        "r2 random 2 1 300 1"
        "r3 random 3 40 1000 60"
        "r4 random 4 300 50 700"
        "r5 random 5 23 2 31"
        "r6 random 6 2 3 8192"
        "r7 random 7 100 5000 100"
        "r8 random 8 1000 20000 1000"
    )
    local spec name args d
    echo "== custom cases: generate and compute C with the starter at one rank (kept under $out_root/custom) =="
    for spec in "${specs[@]}"; do
        read -r name args <<<"$spec"
        if [[ -n "${CUSTOM_CASES:-}" && " $CUSTOM_CASES " != *" $name "* ]]; then continue; fi
        d="$out_root/custom/$name"
        mkdir -p "$d"
        if [[ ! -f "$d/C.csr" || "$script_dir/gen_spgemm.cpp" -nt "$d/C.csr" ]]; then
            local kind rest
            read -r kind rest <<<"$args"
            srun -N1 -n1 "$bin_dir/gen_spgemm" "$kind" "$d" $rest </dev/null
            (ulimit -d 1228800; srun -N1 -n1 -c1 --hint=nomultithread "$orig_dir/hw2-2-seq" "$d/A.csr" "$d/B.csr" "$d/C.csr" \
                </dev/null >"$d/seq.out")
            echo "   $name: starter $(sed -n 's/^Compute time: \([0-9.]*\) s$/\1 s/p' "$d/seq.out")"
        fi
    done
    local s
    for s in "-N1 -n1" "-N1 -n3" "-N2 -n8 --ntasks-per-node=4"; do
        read -r -a shape <<<"$s"
        echo "== custom cases: srun $s =="
        while read -r d; do
            if [[ -n "${CUSTOM_CASES:-}" && " $CUSTOM_CASES " != *" $(basename "$d") "* ]]; then continue; fi
            [[ -n "$d" ]] && run_case "$(basename "$d") $s" "$d" ./hw2-2 "${shape[@]}"
        done < <(custom_dirs)
    done
}

run_sysinfo() {
    echo "== nodes =="
    srun -N2 -n2 --ntasks-per-node=1 bash -c '
        echo "--- $(hostname): $(lscpu | sed -n "s/^Model name: *//p"), $(lscpu | sed -n "s/^NUMA node(s): *//p") NUMA nodes"
        if command -v ibv_devinfo >/dev/null; then
            ibv_devinfo 2>/dev/null | grep -E "hca_id|transport|state|active_mtu|link_layer|active_speed|active_width" || true
        else
            echo "ibv_devinfo not installed"; ls /sys/class/infiniband 2>/dev/null || echo "no /sys/class/infiniband"
        fi
        for n in /sys/class/net/*; do
            [[ -e $n/speed ]] && echo "net $(basename $n): $(cat $n/operstate 2>/dev/null) $(cat $n/speed 2>/dev/null) Mb/s"
        done 2>/dev/null' </dev/null
    echo "== Open MPI transports (pml / btl / mtl components built in) =="
    ompi_info 2>/dev/null | grep -E "MCA (pml|btl|mtl):" || true
}

run_bench() {
    mpicxx "${FLAGS[@]}" -o "$bin_dir/mpi_bench" "$script_dir/mpi_bench.cpp"
    if [[ "$1" == mem ]]; then
        echo "### one node, 8 ranks"
        srun -N1 -n8 -c1 --hint=nomultithread "$bin_dir/mpi_bench" mem </dev/null
    else
        echo "### two nodes, 4 ranks each"
        srun -N2 -n8 --ntasks-per-node=4 -c1 --hint=nomultithread "$bin_dir/mpi_bench" net </dev/null
    fi
}

run_stats() {
    mpicxx "${FLAGS[@]}" -I. -o "$bin_dir/csr_stats" "$script_dir/csr_stats.cpp"
    local c d
    for c in ${CASES:-p01 p02 p03}; do
        d="$(case_dir "$c")"
        echo "########## $c ($d)"
        srun -N1 -n1 "$bin_dir/csr_stats" "$d/A.csr" "$d/B.csr" "$d/C.csr" ${STATS_P:-8 24} </dev/null
    done
}

run_baseline() {
    check_command hw2-2-check
    make -C "$orig_dir" hw2-2-seq
    local c
    echo "== starter on ${CASES:-all public cases}: srun -N2 -n8 --ntasks-per-node=4, 1200 MiB private cap =="
    for c in ${CASES:-$public_cases}; do
        run_case "starter $c" "$(case_dir "$c")" "$orig_dir/hw2-2-seq" -N2 -n8 --ntasks-per-node=4
    done
    echo "(the starter is expected to fail the performance cases)"
}

case "$mode" in
    public) run_public; finish ;;
    ab) run_ab; finish ;;
    ranks) run_ranks; finish ;;
    custom) run_custom; finish ;;
    sysinfo) run_sysinfo ;;
    bench-mem) run_bench mem ;;
    bench-net) run_bench net ;;
    stats) run_stats ;;
    baseline) run_baseline; finish ;;
esac
