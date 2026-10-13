#!/usr/bin/env python3
"""Summarize checked normal runs separately from phase-only diagnostic runs."""
import csv
import re
import statistics
import sys
from pathlib import Path


def write_tsv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, repeat, cases):
    with (root / "runs.tsv").open() as stream:
        runs = list(csv.DictReader(stream, delimiter="\t"))
    phases, counts = [], []
    for run in runs:
        path = root / (run["log"] + ".err")
        text = path.read_text(errors="replace") if path.exists() else ""
        for line in text.splitlines():
            match = re.match(r"\[(hw2-1(?:-light)?)\]\s+(\S+)\s+max\s+([\d.eE+-]+) s\s+min\s+([\d.eE+-]+) s(?:\s+faults (\d+))?", line)
            if match:
                scope, phase, high, low, faults = match.groups()
                phases.append(dict(run, scope=scope, phase=phase, max_s=high, min_s=low, max_faults=faults or ""))
        stats = {}
        for prefix in ("[hw2-1-witness]", "[hw2-1-memory]"):
            for line in text.splitlines():
                if line.startswith(prefix):
                    stats.update(re.findall(r"(\w+)=([^\s]+)", line))
        if stats:
            counts.append(dict(run, **{k: v for k, v in stats.items() if k not in run}))
    keys = ["stage", "case", "variant", "ranks", "repeat", "status"]
    write_tsv(root / "phases.tsv", keys + ["scope", "phase", "max_s", "min_s", "max_faults", "log"], phases)
    write_tsv(root / "counts.tsv", keys + ["tau", "original_edges", "light_edges", "removable_edges", "kept_edges", "retained_fraction", "pivot", "light_reached", "peak_rank_rss_kib", "max_rank_minflt", "arc_counters", "log"], counts)
    lines = ["# HW2-1 Witness Experiment", "", "Normal build only; medians across repetitions on ONE practice node.",
             "These are NOT 24-rank judge/leaderboard times. Do not extrapolate a score from them.", "",
             "| Case | F2 median (s) | Off median (s) | Prune median (s) | F2 / prune |",
             "|---|---:|---:|---:|---:|"]
    variants = ("baseline", "off", "prune")
    totals = dict.fromkeys(variants, 0.0)
    complete = True
    summary = []
    for case in cases:
        medians = {}
        for variant in variants:
            selected = [r for r in runs if r["stage"] == "timing" and r["case"] == case and r["variant"] == variant]
            valid = len(selected) == repeat and all(r["status"] == "PASS" for r in selected)
            if valid:
                values = [float(r["compute_s"]) for r in selected]
                medians[variant] = statistics.median(values)
                totals[variant] += medians[variant]
                summary.append(dict(case=case, variant=variant, runs=len(values), median_s=medians[variant], best_s=min(values), worst_s=max(values), status="PASS"))
            else:
                complete = False
                summary.append(dict(case=case, variant=variant, runs=len(selected), status="INCOMPLETE_OR_FAILED"))
        values = [f"{medians[v]:.6f}" if v in medians else "INCOMPLETE" for v in variants]
        speedup = f"{medians['baseline'] / medians['prune']:.3f}x" if medians.get("prune", 0) > 0 and "baseline" in medians else "-"
        lines.append("| " + " | ".join([case] + values + [speedup]) + " |")
    if complete:
        values = [f"{totals[v]:.6f}" for v in variants]
        speedup = f"{totals['baseline'] / totals['prune']:.3f}x" if totals["prune"] else "-"
        lines.append("| Sum of medians | " + " | ".join(values + [speedup]) + " |")
    else:
        lines += ["", "No total: at least one planned timing run is missing or failed."]
    failures = [r for r in runs if r["status"] != "PASS"]
    profiles = [r for r in runs if r["stage"] == "profile"]
    lines += ["", f"Recorded runs: {len(runs)}; failed: {len(failures)}.",
              f"Diagnostic runs: {len(profiles)} / {len(cases) * 3} planned.", "",
              "## Interpretation", "",
              "- Baseline is the saved F2 revision; off checks whether instrumentation/refactoring changes the normal path.",
              "- Probe builds the light graph and computes pruning decisions, but retains all original non-self edges.",
              "- Prune uses the same preparation and actually removes proven-redundant edges.",
              "- phases.tsv separates main phases from hw2-1-light phases. Main witness already INCLUDES all light phases; never add both.",
              "- gen:filter, trav:* and alloc/free:* are nested measurements, not additional elapsed time.",
              "- Per-phase MPI_MAX values may come from different ranks; their sum is not an exact total.",
              "- counts.tsv gives pruning fractions and peak per-rank RSS. RSS includes shared pages; it is NOT private memory or node-wide RSS.",
              "- Per-arc counters are disabled in phase builds, but profiling still costs time. Only the table above establishes end-to-end speedup.",
              "- Inspect metadata.log, all checker logs and exit-status before accepting a result. Job timeout can leave an incomplete bundle."]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- {r['stage']} {r['case']} {r['variant']} P={r['ranks']}: {r['status']} ({r['log']})" for r in failures]
    write_tsv(root / "summary.tsv", ["case", "variant", "runs", "median_s", "best_s", "worst_s", "status"], summary)
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8 + len(cases) + 2]))


if __name__ == "__main__":
    summarize(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:])
