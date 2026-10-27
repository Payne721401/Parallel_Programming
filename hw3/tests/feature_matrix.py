#!/usr/bin/env python3
"""Small feature matrix. GPU runs require Slurm; generation alone is local-safe."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from case_stats import describe
from run_cases import PROFILE_COLUMNS, PROFILE_TIMES, SPARSE_COLUMNS


def definitions():
    side, seed = 2048, 20261024
    dense = [[0, 0, side, side, 35, seed]]
    def item(name, radius=1, states=2, shape="M", steps=256, rectangles=None):
        centre = int(radius != 1)
        k = ((2*radius+1)**2-1 if shape == "M" else 2*radius*(radius+1)) + centre
        # Larger neighbourhoods use the same count fractions, not public rules.
        slo, shi, blo, bhi = (2, 3, 3, 3) if radius == 1 else (
            math.ceil(k*0.28), math.floor(k*0.48), math.ceil(k*0.28), math.floor(k*0.37))
        return dict(name=name, width=side, height=side, radius=radius, states=states,
                    centre=centre, shape=shape, boundary="torus", steps=steps,
                    rule=f"R{radius},C{states},M{centre},S{slo}..{shi},B{blo}..{bhi},N{shape}",
                    rectangles=dense if rectangles is None else rectangles)
    cluster = [[512+8*x, 512+8*y, 2, 2, 100, seed] for y in range(16) for x in range(16)]
    spread = [[32+128*x, 32+128*y, 2, 2, 100, seed] for y in range(16) for x in range(16)]
    large = item("r1_large")
    large.update(width=8192, height=8192, rectangles=[[0, 0, 8192, 8192, 35, seed]])
    return [item("r1_short", steps=32), item("r1_long"), item("r1_c4", states=4),
            item("moore_r5", radius=5, steps=128), item("moore_r20", radius=20, steps=128),
            item("diamond_r5", radius=5, shape="N", steps=128),
            item("r1_cluster", rectangles=cluster), item("r1_spread", rectangles=spread), large]


def generate(root):
    folder = root / "inputs"
    folder.mkdir(parents=True, exist_ok=False)
    cases = definitions()
    for case in cases:
        text = [f"{case['width']} {case['height']}", case["rule"], case["boundary"],
                str(case["steps"]), str(len(case["rectangles"]))]
        text.extend(" ".join(map(str, rect)) for rect in case["rectangles"])
        path = folder / (case["name"] + ".txt")
        path.write_text("\n".join(text)+"\n", encoding="utf-8")
        describe(path)  # Reject invalid rule/count/rectangle combinations before GPU use.
    (root / "manifest.json").write_text(json.dumps(cases, indent=2)+"\n", encoding="utf-8")
    return cases


def environment(active="auto", reference=False):
    env = os.environ.copy()
    for key in ("PP_KERNEL", "PP_ACTIVE", "PP_R1", "PP_PROFILE", "PP_MOORE", "PP_GRAPH", "PP_EVOLVE", "PP_EVOLVE_EVERY", "PP_SUPPORT", "PP_TEMPORAL"):
        env.pop(key, None)
    env.update(PP_ACTIVE=active, PP_PROFILE="1", PP_SUPPORT="tile", PP_TEMPORAL="1")
    if reference:
        env.update(PP_KERNEL="tiled", PP_MOORE="sat", PP_R1="direct", PP_ACTIVE="0", PP_PROFILE="0")
    return env


def summarize(rows, root):
    extra = ["kernel_interval_ms_per_step", "scheduled_tile_fraction", "kernel_ms_per_scheduled_Mcell"]
    columns = ["name", "schedule", "samples", "steps", "kernel", "compute_mean_s", "compute_min_s",
               "compute_max_s", *PROFILE_TIMES, *extra]
    print("name / schedule: Compute mean [range] s; kernel interval ms/step; scheduled tiles")
    with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for case in definitions():
            for mode in ("auto", "sparse"):
                group = [r for r in rows if r["name"] == case["name"] and r["schedule"] == mode]
                if not group:
                    continue
                if len(group) != 2 or len({r["profile_kernel"] for r in group}) != 1:
                    raise ValueError("missing samples or inconsistent kernel: " + case["name"])
                compute = [float(r["compute_s"]) for r in group]
                out = dict(name=case["name"], schedule=mode, samples=2, steps=case["steps"],
                           kernel=group[0]["profile_kernel"], compute_mean_s=statistics.mean(compute),
                           compute_min_s=min(compute), compute_max_s=max(compute))
                for key in PROFILE_TIMES:
                    out[key] = statistics.mean(float(r[key]) for r in group)
                out[extra[0]] = out["kernels_ms"] / case["steps"]
                if group[0]["sparse_selected"] == "1":
                    if any(tuple(r[k] for k in SPARSE_COLUMNS) != tuple(group[0][k] for k in SPARSE_COLUMNS) for r in group):
                        raise ValueError("inconsistent activity: " + case["name"])
                    ratio = float(group[0]["sparse_mean_work"]) / int(group[0]["sparse_tiles_total"])
                else:
                    ratio = 1.0
                out[extra[1]] = ratio
                cells = case["width"] * case["height"] * case["steps"] * ratio
                out[extra[2]] = out["kernels_ms"] / (cells/1e6) if cells else ""
                writer.writerow(out)
                print(f"{out['name']} / {mode}: {out['compute_mean_s']:.6f} "
                      f"[{min(compute):.6f},{max(compute):.6f}]; "
                      f"{out[extra[0]]:.4f} ms/step; {ratio:.2%}", flush=True)
    print("Kernel interval includes setup/prepare/update/submission gaps; not pure kernel duration.")
    print("All times include profiling. No hidden-case limit or public-score prediction is implied.")


def run(root):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU execution requires srun/sbatch; use --generate-only locally")
    cases = generate(root)
    binary = str(Path("hw3-1").resolve())
    # Full-size cross-check uses the retained dense SAT GPU path, NOT a CPU oracle.
    # Small CPU-reference gates are run by run_features.sh first.
    for case in cases:
        path = root / "inputs" / (case["name"] + ".txt")
        print("Reference dense SAT: " + case["name"], flush=True)
        with (root / (case["name"] + ".reference.log")).open("w", encoding="utf-8") as log:
            subprocess.run([binary, str(path), str(path.with_suffix(".golden.txt"))],
                           env=environment(reference=True), stdout=log, stderr=subprocess.STDOUT,
                           timeout=15, check=True)
    jobs = [(c["name"], "auto") for c in cases]
    jobs += [(name, "sparse") for name in ("r1_cluster", "r1_spread")]
    columns = ["name", "schedule", "round", "case", "verdict", "compute_s", "wall_s", "limit_s",
               *PROFILE_COLUMNS, *SPARSE_COLUMNS]
    rows = []
    with (root / "samples.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for round_id in (1, 2):
            for name, mode in (jobs if round_id == 1 else list(reversed(jobs))):
                out = root / f"{round_id}-{name}-{mode}"
                subprocess.run([sys.executable, "tests/run_cases.py", "--checker", "hw3-1-check",
                                "--wall-timeout", "15", "--outdir", str(out),
                                str(root / "inputs" / (name + ".txt"))],
                               env=environment("1" if mode == "sparse" else "auto"),
                               timeout=30, check=True)
                with (out / "results.tsv").open(encoding="utf-8", newline="") as table:
                    data = list(csv.DictReader(table, delimiter="\t"))
                if len(data) != 1:
                    raise ValueError("expected one case result")
                row = data[0]
                expected_kernel = "sparse_r1" if mode == "sparse" else (
                    "sliding" if name in ("moore_r5", "moore_r20") else
                    "tiled" if name == "diamond_r5" else "dense_r1")
                if row["verdict"] != "AC" or row["profile_status"] != "ok" or row["profile_kernel"] != expected_kernel:
                    raise ValueError("incorrect output, missing profile or wrong dispatch: " + name)
                if mode == "sparse" and (row["sparse_status"] != "ok" or row["sparse_selected"] != "1"):
                    raise ValueError("missing forced sparse diagnostics: " + name)
                row.update(name=name, schedule=mode, round=round_id)
                writer.writerow(row)
                stream.flush()
                rows.append(row)
    summarize(rows, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.generate_only:
            generate(args.outdir)
            print("Generated 9 inputs and manifest; no GPU code executed.")
        else:
            run(args.outdir)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print("feature matrix: " + str(error), file=sys.stderr)
        sys.exit(1)
