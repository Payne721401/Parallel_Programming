#!/usr/bin/env python3
"""Summarize the fixed same-allocation SAT/sliding ABBA; do not promote defaults."""
import csv
import math
from pathlib import Path
import statistics
import sys


def summarize(root, profiled):
    samples = {}
    for round_id, method in ((1, "sat"), (2, "sliding"), (3, "sliding"), (4, "sat")):
        with (root / f"{round_id}-{method}" / "results.tsv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if len(rows) != 2 or {Path(r["case"]).stem for r in rows} != {"05", "09"}:
            raise ValueError("expected exactly cases 05/09 in every round")
        for row in rows:
            if row["verdict"] not in ("AC", "TLE"):
                raise ValueError("unverified output: " + row["case"])
            compute = float(row["compute_s"])
            if not math.isfinite(compute) or compute <= 0:
                raise ValueError("invalid Compute time")
            if profiled:
                allowed = ("tiled", "sparse_tiled") if method == "sat" else ("sliding", "sparse_sliding")
                if row["profile_status"] != "ok" or row["profile_kernel"] not in allowed:
                    raise ValueError("missing or unexpected kernel profile: " + row["case"])
                if row["sparse_status"] not in ("absent", "ok"):
                    raise ValueError("invalid sparse diagnostics")
            samples.setdefault((Path(row["case"]).stem, method), []).append(row)
    print("Case  SAT mean(s)  Sliding mean(s)  SAT/Sliding  measured ranges(s)")
    for case in ("05", "09"):
        a = samples[case, "sat"]
        b = samples[case, "sliding"]
        ta = [float(r["compute_s"]) for r in a]
        tb = [float(r["compute_s"]) for r in b]
        ma, mb = statistics.mean(ta), statistics.mean(tb)
        print(f"{case}    {ma:.6f}     {mb:.6f}         {ma/mb:.3f}x"
              f"       SAT {min(ta):.6f}..{max(ta):.6f}, sliding {min(tb):.6f}..{max(tb):.6f}")
        if profiled:
            ka = statistics.mean(float(r["kernels_ms"]) for r in a)
            kb = statistics.mean(float(r["kernels_ms"]) for r in b)
            print(f"  kernel interval mean: SAT={ka:.3f} ms sliding={kb:.3f} ms")
            # Same input/rules must produce the same work counts. Different queue
            # order is permitted, but not different activity or fallback counts.
            keys = ("sparse_status", "sparse_selected", "sparse_tiles_total", "sparse_initial_work",
                    "sparse_mean_work", "sparse_max_work", "sparse_dense_steps")
            if len({tuple(r[k] for k in keys) for r in a + b}) != 1:
                raise ValueError("sparse work differs between runs: case " + case)
    print("Two samples per method; small differences can be noise. Profiling=" + str(int(profiled)))


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3 or sys.argv[2] not in ("0", "1"):
            raise ValueError("usage: summarize_sliding.py LOGDIR PROFILE_0_OR_1")
        summarize(Path(sys.argv[1]), sys.argv[2] == "1")
    except (OSError, ValueError, KeyError) as error:
        print("sliding summary: " + str(error), file=sys.stderr)
        sys.exit(1)
