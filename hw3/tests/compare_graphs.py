#!/usr/bin/env python3
"""ABBA graph comparison using retained feature inputs/goldens; run under Slurm."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from feature_matrix import environment
from run_cases import SPARSE_COLUMNS

JOBS = [("r1_short", "auto"), ("r1_long", "auto"), ("r1_c4", "auto"),
        ("r1_large", "auto"), ("r1_cluster", "1"), ("moore_r5", "auto")]


def graph_diagnostic(text, requested, steps):
    lines = [line for line in text.splitlines() if line.startswith("[PP_GRAPH] ")]
    if len(lines) != 1:
        raise ValueError("missing/duplicate graph diagnostic; rebuild the binary")
    fields = dict(token.split("=", 1) for token in lines[0].split()[1:])
    expected = dict(requested=requested, selected=requested, batch=32,
                    replays=steps//32 if requested else 0,
                    tail=steps%32 if requested else steps)
    if any(int(fields[key]) != value for key, value in expected.items()):
        raise ValueError("unexpected graph dispatch or replay/tail count")
    build = float(fields["build_host_ms"])
    if not math.isfinite(build) or build < 0:
        raise ValueError("invalid graph build duration")
    return build


def summarize(rows, root):
    fields = ["name", "active", "off_mean_s", "on_mean_s", "off_over_on",
              "off_min_s", "off_max_s", "on_min_s", "on_max_s", "on_build_mean_ms"]
    with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        print("Input / active: off -> graph (mean seconds); speedup; sample ranges")
        for name, active in JOBS:
            groups = [[r for r in rows if r["name"] == name and r["graph"] == str(mode)] for mode in (0, 1)]
            if any(len(g) != 2 for g in groups):
                raise ValueError("expected two samples per method: " + name)
            all_rows = groups[0] + groups[1]
            if len({r["profile_kernel"] for r in all_rows}) != 1:
                raise ValueError("kernel changed between methods: " + name)
            if len({tuple(r[k] for k in SPARSE_COLUMNS) for r in all_rows}) != 1:
                raise ValueError("activity changed between methods: " + name)
            values = [[float(r["compute_s"]) for r in group] for group in groups]
            off, on = map(statistics.mean, values)
            build = [float(r["graph_build_ms"]) for r in groups[1] if r["graph_build_ms"] != ""]
            row = dict(name=name, active=active, off_mean_s=off, on_mean_s=on,
                       off_over_on=off/on, off_min_s=min(values[0]), off_max_s=max(values[0]),
                       on_min_s=min(values[1]), on_max_s=max(values[1]),
                       on_build_mean_ms=statistics.mean(build) if build else "")
            writer.writerow(row)
            print(f"{name} / {active}: {off:.6f} -> {on:.6f}; {off/on:.3f}x; "
                  f"off {min(values[0]):.6f}..{max(values[0]):.6f}, "
                  f"on {min(values[1]):.6f}..{max(values[1]):.6f}; build_ms={row['on_build_mean_ms']}")
    print("Compute includes graph construction, first launch, tail, waits and destruction.")
    print("Build time is nested within the kernel interval; do not add the two. No defaults changed.")


def run(matrix, root, profile):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU execution requires srun/sbatch")
    cases = {c["name"]: c for c in json.loads((matrix / "manifest.json").read_text(encoding="utf-8"))}
    for name, _ in JOBS:
        if cases[name]["steps"] < 32:
            raise ValueError("comparison requires at least one graph batch")
        for suffix in ("txt", "golden.txt"):
            if not (matrix / "inputs" / f"{name}.{suffix}").is_file():
                raise ValueError("missing matrix input/golden: " + name)
    rows = []
    with (root / "samples.tsv").open("x", encoding="utf-8", newline="") as stream:
        writer = None
        for round_id, method in enumerate((0, 1, 1, 0), 1):
            for name, active in JOBS:
                out = root / f"{round_id}-{method}-{name}"
                env = environment(active)
                env.update(PP_GRAPH=str(method), PP_PROFILE=str(profile))
                subprocess.run([sys.executable, "tests/run_cases.py", "--checker", "hw3-1-check",
                                "--wall-timeout", "15", "--outdir", str(out),
                                str(matrix / "inputs" / f"{name}.txt")],
                               env=env, timeout=30, check=True)
                with (out / "results.tsv").open(encoding="utf-8", newline="") as table:
                    data = list(csv.DictReader(table, delimiter="\t"))
                if len(data) != 1 or data[0]["verdict"] != "AC":
                    raise ValueError("invalid case result: " + name)
                row = data[0]
                build = ""
                if profile:
                    expected = "sparse_r1" if active == "1" else "sliding" if name == "moore_r5" else "dense_r1"
                    if row["profile_status"] != "ok" or row["profile_kernel"] != expected:
                        raise ValueError("missing profile or unexpected kernel: " + name)
                    if active == "1" and (row["sparse_status"] != "ok" or row["sparse_selected"] != "1"):
                        raise ValueError("missing sparse statistics: " + name)
                    build = graph_diagnostic((out / f"00-{name}.log").read_text(encoding="utf-8"),
                                             method, cases[name]["steps"])
                row.update(name=name, active=active, graph=str(method), round=round_id, graph_build_ms=build)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row), delimiter="\t")
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
    summarize(rows, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--profile", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    try:
        run(args.matrix, args.outdir, args.profile)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print("graph comparison: " + str(error), file=sys.stderr)
        sys.exit(1)
