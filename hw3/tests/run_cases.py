#!/usr/bin/env python3
"""Run selected official-format cases; keep wall timeout separate from Compute time.

On Nano5 execute this script inside srun/sbatch. Defaults to the six examples.
No default public sweep: the naive kernel can be very expensive on large cases.
"""
import argparse
import csv
import datetime
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


PROFILE_TIMES = ["alloc_host_ms", "h2d_ms", "kernels_ms", "d2h_ms", "free_host_ms", "run_host_ms"]
PROFILE_COLUMNS = ["profile_status", "profile_kernel", *PROFILE_TIMES]
SPARSE_COLUMNS = ["sparse_status", "sparse_selected", "sparse_tiles_total", "sparse_initial_work",
                  "sparse_mean_work", "sparse_max_work", "sparse_dense_steps"]


def parse_profile(log_text):
    lines = [line for line in log_text.splitlines() if line.startswith("[PP_PROFILE] ")]
    if not lines:
        return ["absent", "", *([""] * len(PROFILE_TIMES))]
    try:
        if len(lines) != 1:
            raise ValueError("expected one profile per process")
        fields = dict(token.split("=", 1) for token in lines[0].split()[1:])
        if fields["kernel"] not in ("naive", "tiled", "sparse_naive", "sparse_tiled", "none"):
            raise ValueError("invalid profile kernel")
        times = [float(fields[name]) for name in PROFILE_TIMES]
        if any(not math.isfinite(value) or value < 0 for value in times):
            raise ValueError("invalid profile duration")
        return ["ok", fields["kernel"], *times]
    except (KeyError, ValueError):
        # Preserve correctness/Compute time results, but do not present malformed
        # diagnostic values as valid measurements. The original log is retained.
        return ["invalid", "", *([""] * len(PROFILE_TIMES))]


def parse_sparse(log_text):
    lines = [line for line in log_text.splitlines() if line.startswith("[PP_SPARSE] ")]
    empty = [""] * (len(SPARSE_COLUMNS) - 1)
    if not lines:
        return ["absent", *empty]
    try:
        if len(lines) != 1:
            raise ValueError("expected one sparse diagnostic")
        fields = dict(token.split("=", 1) for token in lines[0].split()[1:])
        selected, total, initial, work_sum, maximum, dense, steps = (
            int(fields[key]) for key in ("selected", "tiles_total", "initial_work", "work_sum",
                                        "work_max", "dense_steps", "steps"))
        if (selected not in (0, 1) or total <= 0 or steps <= 0 or not 0 <= initial <= total
                or not 0 <= maximum <= total or not 0 <= work_sum <= total * steps
                or not 0 <= dense <= steps or (not selected and (work_sum or maximum or dense))):
            raise ValueError("invalid sparse counts")
        return ["ok", selected, total, initial, work_sum / steps, maximum, dense]
    except (KeyError, ValueError):
        return ["invalid", *empty]


def run_cases(args):
    paths = [Path(p) for p in args.inputs] if args.inputs else sorted(Path("examples").rglob("*.txt"))
    if not paths:
        raise ValueError("no input cases found; run from hw3/")
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise ValueError(f"binary missing: {binary}")
    checker = shutil.which(args.checker) if args.checker else None
    if args.checker and not checker:
        raise ValueError(f"checker not found: {args.checker}")
    cases = []
    for path in paths:
        golden = path.with_suffix(".golden.txt")
        if not golden.is_file():
            golden = path.with_suffix(".out")
        if not path.is_file() or not golden.is_file():
            raise ValueError(f"input or golden missing: {path} / {golden}")
        cases.append((path, golden))

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    outdir = Path(args.outdir or f"tmp/baseline-{stamp}")
    outdir.mkdir(parents=True, exist_ok=False)
    failed = 0
    with (outdir / "results.tsv").open("w", newline="", encoding="utf-8") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["case", "verdict", "compute_s", "wall_s", "limit_s", *PROFILE_COLUMNS, *SPARSE_COLUMNS])
        for number, (path, golden) in enumerate(cases):
            stem = f"{number:02d}-{path.stem}"
            output = outdir / f"{stem}.out"
            log = outdir / f"{stem}.log"
            start = time.monotonic()
            compute = None
            verdict = "RE"
            limit = args.limit
            # Infer published limits only for paths in the official cases directory.
            if limit is None and path.parent.name == "cases" and re.fullmatch(r"0[1-9]|1[0-2]", path.stem):
                case_id = int(path.stem)
                limit = .2 if case_id <= 4 else .5 if case_id <= 7 else .7 if case_id <= 10 else .8
            with log.open("w", encoding="utf-8") as stream:
                try:
                    result = subprocess.run([str(binary), str(path), str(output)],
                                            stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=args.wall_timeout, check=False)
                    verdict = "PENDING" if result.returncode == 0 else "RE"
                except subprocess.TimeoutExpired:
                    verdict = "WALL_TIMEOUT"
            wall = time.monotonic() - start
            log_text = log.read_text(encoding="utf-8", errors="replace")
            profile = parse_profile(log_text)
            sparse = parse_sparse(log_text)
            match = re.search(r"Compute time:\s*([0-9.eE+-]+)\s*s", log_text)
            if match:
                try:
                    compute = float(match.group(1))
                except ValueError:
                    pass
            if verdict == "PENDING":
                if not output.is_file():
                    verdict = "NO"
                elif compute is None or not math.isfinite(compute) or compute < 0:
                    verdict = "BAD_TIMING"
                else:
                    if checker:
                        with (outdir / f"{stem}.check.log").open("w", encoding="utf-8") as stream:
                            try:
                                checked = subprocess.run([checker, str(golden), str(output)], stdout=stream,
                                                         stderr=subprocess.STDOUT, timeout=10, check=False)
                                same = checked.returncode == 0
                            except subprocess.TimeoutExpired:
                                verdict = "CHECK_TIMEOUT"
                                same = False
                    else:
                        same = output.read_bytes().split() == golden.read_bytes().split()
                    if verdict == "PENDING":
                        verdict = "WA" if not same else "TLE" if limit is not None and compute > limit else "AC"
            failed += verdict != "AC"
            writer.writerow([str(path), verdict, "" if compute is None else compute, f"{wall:.6f}", limit or "", *profile, *sparse])
            table.flush()
            print(f"{path}: {verdict} compute={compute} wall={wall:.3f}s limit={limit}", flush=True)
            if profile[0] != "absent":
                print("  " + " ".join(f"{key}={value}" for key, value in zip(PROFILE_COLUMNS, profile)
                                       if value != ""), flush=True)
            if sparse[0] != "absent":
                print("  " + " ".join(f"{key}={value}" for key, value in zip(SPARSE_COLUMNS, sparse)
                                       if value != ""), flush=True)
    print(f"PASS {len(cases)-failed}/{len(cases)}; logs: {outdir}")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--binary", default="./hw3-1")
    parser.add_argument("--checker", help="optional official checker, e.g. hw3-1-check")
    parser.add_argument("--wall-timeout", type=float, default=20,
                        help="whole-process safety timeout, NOT the Compute time limit")
    parser.add_argument("--limit", type=float, help="Compute time limit in seconds")
    parser.add_argument("--outdir", help="new output directory; refuses to overwrite an existing run")
    args = parser.parse_args()
    if not math.isfinite(args.wall_timeout) or args.wall_timeout <= 0:
        parser.error("--wall-timeout must be finite and positive")
    if args.limit is not None and (not math.isfinite(args.limit) or args.limit <= 0):
        parser.error("--limit must be finite and positive")
    try:
        return run_cases(args)
    except (OSError, ValueError) as error:
        print(f"run_cases: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
