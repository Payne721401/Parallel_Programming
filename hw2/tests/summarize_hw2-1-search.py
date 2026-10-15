#!/usr/bin/env python3
"""Keep normal timings, diagnostic counts and invariant checks separate."""
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


def summarize(root, mode, repeat, cases):
    with (root / "runs.tsv").open() as stream:
        runs = list(csv.DictReader(stream, delimiter="\t"))
    counts, phases, witnesses, checks = [], [], [], []
    errors = []
    for run in runs:
        path = root / (run["log"] + ".err")
        text = path.read_text(errors="replace") if path.exists() else ""
        scopes = {}
        for line in text.splitlines():
            match = re.match(r"\[(hw2-1(?:-light)?)\]\s+(\S+)\s+sum (\d+)\s+max/rank (\d+)", line)
            if match:
                scope, name, total, high = match.groups()
                counts.append(dict(run, scope=scope, name=name, sum=total, max_rank=high))
                scopes.setdefault(scope, {})[name] = int(total)
            match = re.match(r"\[(hw2-1(?:-light)?)\]\s+(\S+)\s+max\s+([\d.eE+-]+) s\s+min\s+([\d.eE+-]+) s", line)
            if match:
                scope, name, high, low = match.groups()
                phases.append(dict(run, scope=scope, name=name, max_s=high, min_s=low))
            if line.startswith("[hw2-1-witness]") or line.startswith("[hw2-1-memory]"):
                witnesses.append(dict(run, **dict(re.findall(r"(\w+)=([^\s]+)", line))))
        if run["variant"] not in ("bucket", "atomic") or run["status"] != "PASS":
            continue
        # Main search can be absent for isolated sources; light search can be absent
        # for an empty witness graph. Every executed search reports pushes, even zero.
        for scope, c in scopes.items():
            if "pushes" not in c:
                continue
            required = {"bucket-seeds", "bucket-stale", "bucket-published", "bucket-inspected"}
            ok = required <= c.keys()
            if ok:
                ok = c["pushes"] + c["bucket-seeds"] == c["bucket-stale"] + c["bucket-published"]
                ok &= c["bucket-inspected"] >= c["bucket-stale"] + c["bucket-published"]
            if run["variant"] == "atomic":
                required = {"cas-attempts", "cas-failed", "cas-changed", "cas-skipped", "cas-abandoned", "arcs"}
                ok &= required <= c.keys()
                if required <= c.keys():
                    ok &= c["cas-attempts"] == c["pushes"] + c["cas-failed"]
                    ok &= c["cas-skipped"] + c["pushes"] + c["cas-abandoned"] == c["arcs"]
                    ok &= c["cas-changed"] <= c["cas-failed"]
            checks.append(dict(run, scope=scope, invariant="PASS" if ok else "FAIL"))
            if not ok:
                errors.append(f"Counter invariant failed: {run['log']} {scope}")
    keys = ["stage", "case", "variant", "ranks", "repeat", "divisor", "status", "log"]
    write_tsv(root / "counts.tsv", keys + ["scope", "name", "sum", "max_rank"], counts)
    write_tsv(root / "phases.tsv", keys + ["scope", "name", "max_s", "min_s"], phases)
    write_tsv(root / "invariants.tsv", keys + ["scope", "invariant"], checks)
    fields = sorted({k for row in witnesses for k in row} - set(keys))
    write_tsv(root / "witness.tsv", keys + fields, witnesses)
    lines = ["# HW2-1 Search Experiments", "", "One practice allocation, not 24-rank judge timings.",
             "Only normal-build medians below measure end-to-end performance.", ""]
    if mode != "bucket":
        lines += ["| Case | Full witness | Half witness | Quarter witness | Half change | Quarter change |",
                  "|---|---:|---:|---:|---:|---:|"]
        for case in cases:
            med = {}
            for d in ("1", "2", "4"):
                selected = [r for r in runs if r["stage"] == "timing" and r["case"] == case and r["divisor"] == d]
                if len(selected) == repeat and all(r["status"] == "PASS" for r in selected):
                    med[d] = statistics.median(float(r["compute_s"]) for r in selected)
            values = [f"{med[d]:.6f}" if d in med else "INCOMPLETE" for d in ("1", "2", "4")]
            changes = [f"{100 * (med[d] / med['1'] - 1):+.1f}%" if d in med and med.get("1", 0) > 0 else "-" for d in ("2", "4")]
            lines.append("| " + " | ".join([case] + values + changes) + " |")
            if len(med) != 3:
                errors.append(f"Incomplete normal timing set: {case}")
    expected = len(cases) * {"all": 5, "sample": 3, "bucket": 3}[mode]
    profiles = sum(r["stage"] == "profile" for r in runs)
    if profiles != expected:
        errors.append(f"Incomplete diagnostic set: {profiles}/{expected}")
    failures = sum(r["status"] != "PASS" for r in runs)
    if failures:
        errors.append(f"Checker/runtime failures: {failures}")
    lines += ["", f"Checked runs: {len(runs)}; failed: {failures}.",
              f"Diagnostic runs: {profiles}/{expected}; counter checks: {len(checks)}.",
              "", "## Interpretation", "",
              "- runs.tsv preserves every run, including failures; compare variation, not just medians.",
              "- phase has no per-arc or per-push diagnostics; bucket adds queue counters and growth-event timing; atomic also counts CAS attempts/failures. Do not mix their timings.",
              "- counts.tsv separates hw2-1 (main) from hw2-1-light (witness). Each row gives a sum and a per-rank maximum.",
              "- bucket-inspected counts visits, not unique entries; inspected-stale-published counts repeated valid inspections.",
              "- bucket-stale/(pushes+bucket-seeds) is the stale-entry fraction. bucket-moved-bytes counts elements copied on vector growth, not all memory traffic.",
              "- bucket:grow is nested inside search, mainly trav:relax; it includes timing overhead and must not be added to those times.",
              "- cas-failed/cas-attempts is the failure rate; cas-changed is evidence of concurrent decreases. Failures with unchanged expected values may be spurious.",
              "- Low CAS failure rates do not rule out cache-line contention, false sharing or NUMA cost. These counters do not measure those costs.",
              "- phases.tsv contains nested MPI_MAX timings; maxima may belong to different ranks. Main witness already includes light phases.",
              "- witness.tsv records sample_divisor, retained edges and peak rank RSS. RSS includes shared pages, not private-only or node-wide usage.",
              "- Require exit-status=0, all checker passes and complete diagnostics before accepting the experiment."]
    if errors:
        lines += ["", "## Incomplete Or Failed", ""] + ["- " + e for e in errors]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(summarize(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4:]))
