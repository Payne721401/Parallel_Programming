#!/usr/bin/env python3
"""Measure existing defaults before architecture changes; never report a speedup.

timing: three unprofiled repetitions, then one phase/dispatch observation.
ncu: one matching evolution launch at mid-run and near the end, full output checked.
The whole simulation is executed for EACH capture. Profiler times are diagnostic only.
"""
import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

from run_cases import PROFILE_TIMES, run_cases
from summarize_bitplane import records, verify_public_dispatch

GROUPS = {"sparse": ("08", "09", "10"), "r1": ("11", "12"),
          "large": ("05", "06", "07")}
SECTIONS = ("SpeedOfLight", "LaunchStats", "Occupancy", "MemoryWorkloadAnalysis",
            "SchedulerStats", "WarpStateStats", "ComputeWorkloadAnalysis",
            "InstructionStats", "SourceCounters")
KERNEL_FILTER = r"step_(r1_(dense|bits)|sparse|tiled|naive)"
METRICS = {"duration_us": "Duration", "sm_pct": "Compute (SM) Throughput",
           "dram_pct": "DRAM Throughput", "occupancy_pct": "Achieved Occupancy",
           "registers": "Registers Per Thread", "no_eligible_pct": "No Eligible",
           "eligible_warps": "Eligible Warps Per Scheduler"}


def windows(steps):
    if steps < 4:
        raise ValueError("phase sampling requires at least four generations")
    # launch-skip counts matching step kernels only, not init/pack/prepare kernels.
    return (("middle", steps // 2), ("late", steps - 3))


def numeric(text):
    value = float(text.replace(",", ""))
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid metric: " + text)
    return value


def parse_ncu(text):
    """Keep section/unit information; fail on missing/extra evolution launches."""
    launches, metrics = [], []
    section, in_table = "", False
    for line in text.splitlines():
        if re.search(KERNEL_FILTER, line) and "Context " in line and "Stream " in line:
            launches.append(line.strip())
        if line.strip().startswith("Section: "):
            section = line.strip()[9:]
        if line.strip().startswith("Metric Name"):
            in_table = True
            continue
        if not line.strip():
            in_table = False
        if not in_table or line.lstrip().startswith("---"):
            continue
        fields = re.split(r"\s{2,}", line.strip())
        if len(fields) in (2, 3):
            name, value = fields[0], fields[-1]
            metrics.append(dict(section=section, metric=name,
                                unit=fields[1] if len(fields) == 3 else "", value=value))
    if len(launches) != 1:
        raise ValueError("expected ONE evolution launch in ncu report, got " + str(len(launches)))
    result = {"kernel": launches[0]}
    for key, name in METRICS.items():
        matches = [m for m in metrics if m["metric"] == name]
        if len(matches) != 1:
            raise ValueError("missing/duplicate ncu metric: " + name)
        item = matches[0]
        value = numeric(item["value"])
        if key == "duration_us":
            scales = {"nsecond": .001, "usecond": 1, "msecond": 1000, "second": 1000000}
            if item["unit"] not in scales:
                raise ValueError("unknown duration unit: " + item["unit"])
            value *= scales[item["unit"]]
        result[key] = value
    return result, metrics


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def command(argv, log, timeout=150):
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode:
        raise ValueError("command failed; inspect " + str(log))


def timing(cases, root):
    samples, phases = [], {}
    for repetition in range(1, 5):
        profiled = repetition == 4
        os.environ["PP_PROFILE"] = str(int(profiled))
        order = cases if repetition % 2 else tuple(reversed(cases))
        folder = root / ("profile" if profiled else "timing-" + str(repetition))
        print("Phase observation" if profiled else "Unprofiled repetition " + str(repetition))
        args = argparse.Namespace(inputs=["cases/" + c + ".txt" for c in order],
                                  binary="./hw3-1", checker="hw3-1-check",
                                  wall_timeout=20, limit=None, outdir=str(folder))
        status = run_cases(args)
        with (folder / "results.tsv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if status not in (0, 1) or len(rows) != len(cases):
            raise ValueError("incomplete timing run")
        for index, row in enumerate(rows):
            case = Path(row["case"]).stem
            if row["verdict"] not in ("AC", "TLE"):
                raise ValueError("output not verified correct: " + case)
            numeric(row["compute_s"])
            if profiled:
                log = (folder / (f"{index:02d}-{case}.log")).read_text(encoding="utf-8")
                data = records(log, "PP_BITPLANE")
                if len(data) != 1:
                    raise ValueError("missing/duplicate bit-plane dispatch; rebuild binary")
                verify_public_dispatch(data[0], row, Path(row["case"]).read_text().splitlines(),
                                       "auto", 4, "compact")
                phases[case] = row
            elif row["profile_status"] != "absent":
                raise ValueError("profiling unexpectedly enabled during timing")
            samples.append(dict(repetition=repetition, profiled=int(profiled), **row))
    write_table(root / "timing-samples.tsv", samples)
    summary = []
    for case in cases:
        values = [float(s["compute_s"]) for s in samples if Path(s["case"]).stem == case and not s["profiled"]]
        low, high = min(values), max(values)
        row = dict(case=case, samples=len(values), mean_s=statistics.mean(values),
                   median_s=statistics.median(values), min_s=low, max_s=high,
                   timing_status="unstable_samples" if low == 0 or high > 1.2 * low else "ok",
                   profile_kernel=phases[case]["profile_kernel"],
                   profile_compute_s=phases[case]["compute_s"])
        row.update({k: phases[case][k] for k in PROFILE_TIMES})
        row.update({k: phases[case][k] for k in ("sparse_mean_work", "sparse_max_work", "sparse_dense_steps")})
        summary.append(row)
        print(f"{case}: median={row['median_s']:.6f}s range={low:.6f}..{high:.6f} "
              f"{row['timing_status']} kernel={row['profile_kernel']}")
    write_table(root / "timing-summary.tsv", summary)
    print("Same configuration repeated, NOT an A/B speedup. Profile observation is separate.")


def ncu(cases, root):
    os.environ["PP_PROFILE"] = "0"
    folder = root / "ncu"
    folder.mkdir()
    command(["ncu", "--version"], folder / "version.log", 15)
    command(["ncu", "--list-sections"], folder / "sections.log", 30)
    available = (folder / "sections.log").read_text(encoding="utf-8", errors="replace")
    if any(not re.search(r"\b" + s + r"\b", available) for s in SECTIONS):
        raise ValueError("ncu lacks a requested section; inspect ncu/sections.log")
    summaries, all_metrics = [], []
    for case in cases:
        path = Path("cases/" + case + ".txt")
        steps = int(path.read_text().splitlines()[3])
        for phase, skip in windows(steps):
            prefix = folder / (case + "-" + phase)
            print(f"Diagnostic ncu {case} {phase}: generation={skip+1}/{steps}, one launch; NOT judge timing")
            argv = ["ncu", "--kernel-name-base", "demangled", "--kernel-name", "regex:" + KERNEL_FILTER,
                    "--launch-skip", str(skip), "--launch-count", "1"]
            for section in SECTIONS:
                argv.extend(["--section", section])
            argv.extend(["--export", str(prefix), "./hw3-1", str(path), str(prefix) + ".out"])
            command(argv, Path(str(prefix) + ".capture.log"))
            command(["hw3-1-check", str(path.with_suffix(".golden.txt")), str(prefix) + ".out"],
                    Path(str(prefix) + ".check.log"), 10)
            stats = Path(str(prefix) + ".stats.txt")
            command(["ncu", "--import", str(prefix) + ".ncu-rep", "--page", "details"], stats, 60)
            result, metrics = parse_ncu(stats.read_text(encoding="utf-8", errors="replace"))
            context = dict(case=case, phase=phase, generation=skip+1, steps=steps, output_verified=1)
            summaries.append(dict(**context, **result))
            all_metrics.extend(dict(**context, **m) for m in metrics)
            # Flush after each checked capture: partial results survive a disconnected/killed job.
            write_table(root / "ncu-summary.tsv", summaries)
            write_table(root / "ncu-metrics.tsv", all_metrics)
            print("  verified; duration={duration_us:.2f}us SM={sm_pct:.2f}% DRAM={dram_pct:.2f}% "
                  "occupancy={occupancy_pct:.2f}% no_eligible={no_eligible_pct:.2f}% regs={registers:g}".format(**result))
    print("Raw instruction/stall details: ncu/*.stats.txt; all parsed metrics: ncu-metrics.tsv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--mode", choices=("all", "timing", "ncu", "plan"), default="all")
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    cases = GROUPS[args.group]
    if args.mode == "plan":
        print("Cases: " + " ".join(cases))
        print(f"Timing: {len(cases)*3} unprofiled + {len(cases)} phase observations, all checked.")
        print(f"NCU: {len(cases)*2} full simulations, ONE evolution kernel each, middle and late.")
        print("Kernel/default changes: NONE. No GPU work performed by plan mode.")
        return
    if not os.environ.get("SLURM_JOB_ID") or args.outdir is None:
        raise ValueError("use run_arch_measure.sh through srun/sbatch")
    for tool in ("hw3-1-check", "nvidia-smi", "git") + (("ncu",) if args.mode != "timing" else ()):
        if not shutil.which(tool):
            raise ValueError("missing command: " + tool)
    for case in cases:
        for suffix in (".txt", ".golden.txt"):
            if not Path("cases/" + case + suffix).is_file():
                raise ValueError("missing input/golden for " + case)
    if not Path("hw3-1").is_file():
        raise ValueError("run make hw3-1 first")
    # Clear inherited experiment knobs, including future PP_* additions.
    for name in list(os.environ):
        if name.startswith("PP_"):
            del os.environ[name]
    os.environ.update(PP_GRAPH="0", PP_EVOLVE="0", PP_PROFILE="0",
                      PP_BITPLANE="auto", PP_BIT_ROWS="4", PP_BIT_LOGIC="compact")
    root = args.outdir
    root.mkdir(parents=True, exist_ok=True)
    # Refuse an accidental rerun into an existing result directory.
    with (root / "measurement-started").open("x") as stream:
        stream.write(args.group + " " + args.mode + "\n")
    command(["git", "rev-parse", "HEAD"], root / "revision.log", 10)
    command(["git", "diff", "--", "hw3-1.cu", "r1_bits.hpp", "r1_bitplane.cuh", "tests"], root / "working-diff.log", 10)
    print("Revision: " + (root / "revision.log").read_text().strip())
    print("CUDA_MODULE_LOADING=" + os.environ.get("CUDA_MODULE_LOADING", "unset"))
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], check=True, timeout=15)
    hashes = []
    for path in [Path("hw3-1"), *sorted(Path(".").glob("*.cu")), *sorted(Path(".").glob("*.cuh")),
                 *sorted(Path(".").glob("*.hpp")), Path(__file__), Path("tests/run_arch_measure.sh")]:
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path))
    (root / "sha256.log").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    if args.mode in ("all", "timing"):
        timing(cases, root)
    if args.mode in ("all", "ncu"):
        ncu(cases, root)
    (root / "COMPLETE").write_text("All requested measurements and output checks completed.\n")
    print("Complete: " + str(root) + "; no production defaults changed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print("arch measurement failed: " + str(error), file=sys.stderr)
        sys.exit(1)
