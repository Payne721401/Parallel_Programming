#!/usr/bin/env bash
set -euo pipefail

# HW 2-2 measurements, for SEIREN. Run from anywhere: the script works in hw2/hw2-2. It must run
# inside one allocation (every srun a job step); hand it to salloc, which the practice QoS `hw`
# limits to 16 CPUs (8 ranks with nomultithread), 32G and 2 nodes. Two shapes:
#
#   salloc -N1 -n8 -c1 --hint=nomultithread --mem=32G -t 20 bash ../tests/run_hw2-2.sh bench-mem
#   salloc -N2 -n8 --ntasks-per-node=4 -c1 --hint=nomultithread --mem=16G -t 40 bash ../tests/run_hw2-2.sh <mode>
#
# (--mem is per node.) Modes:
#   bench-mem tests/mpi_bench mem, 8 ranks on one node (the first shape): first touch and
#             release of private vs shared memory with 1, 2, 4, 8 ranks at once; copy bandwidth
#   bench-net tests/mpi_bench net, 4 ranks on each of two nodes: latency and bandwidth within a
#             node and across nodes, all pairs streaming across at once, MPI_Alltoall
#   sysinfo   both nodes: CPU, NUMA, network devices and their state, Open MPI's transports
#   stats     tests/csr_stats on CASES (default p01 p02 p03): sizes, work per row, rank and node
#             under row-wise and K-split distributions, cross-node traffic, partial-C sizes,
#             local kernel speed. Serial and uncapped (it holds a whole B)
#   baseline  the starter (hw2-2-seq) on CASES (default every public case) in the practice shape
#             -N2 -n8, under the judge's 1200 MiB private cap, checked with hw2-2-check
#
# Env overrides: CASES="c01 p01", CASES_DIR, OUT_ROOT, STATS_P="8 24" (rank counts for stats).

mode="${1:-}"
case "$mode" in
    sysinfo|bench-mem|bench-net|stats|baseline) ;;
    *)
        echo "usage: $0 sysinfo|bench-mem|bench-net|stats|baseline" >&2
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
    if ! command -v hw2-2-check >/dev/null 2>&1; then
        echo "missing command: hw2-2-check" >&2
        exit 1
    fi
    make -C "$orig_dir" hw2-2-seq
    local c d st t rss passed=0 failed=0
    local shape=(srun -N2 -n8 --ntasks-per-node=4 -c1 --hint=nomultithread)
    echo "== starter on ${CASES:-all public cases}: ${shape[*]}, 1200 MiB private cap =="
    printf '%-6s %-5s %-14s %s\n' case stat compute peak_rss
    for c in ${CASES:-c01 c02 c03 c04 c05 c06 c07 c08 c09 c10 c11 c12 c13 c14 p01 p02 p03}; do
        d="$(case_dir "$c")"
        local log="$out_root/baseline.$c"
        rm -f "$out_root/out.csr"
        if (ulimit -d 1228800; "${shape[@]}" "$orig_dir/hw2-2-seq" "$d/A.csr" "$d/B.csr" "$out_root/out.csr" \
                </dev/null >"$log.out" 2>"$log.err"); then
            if hw2-2-check "$d/C.csr" "$out_root/out.csr" >"$log.check" 2>&1; then st=PASS; else st=WA; fi
        else
            st=RE
        fi
        t="$(sed -n 's/^Compute time: \([0-9.]*\) s$/\1/p' "$log.out")"
        rss="$(sed -n 's/^Peak RSS: \([0-9]*\) KiB$/\1/p' "$log.out")"
        printf '%-6s %-5s %-14s %s\n' "$c" "$st" "${t:--}" "${rss:+$((rss / 1024)) MiB}"
        if [[ "$st" == PASS ]]; then passed=$((passed + 1)); else
            failed=$((failed + 1))
            grep -h -m2 -E "rank [0-9]+:|bad_alloc|error" "$log.err" "$log.out" 2>/dev/null | sed 's/^/       /' || true
        fi
    done
    echo "== $passed passed, $failed failed (the starter is expected to fail the performance cases) =="
}

case "$mode" in
    sysinfo) run_sysinfo ;;
    bench-mem) run_bench mem ;;
    bench-net) run_bench net ;;
    stats) run_stats ;;
    baseline) run_baseline ;;
esac
