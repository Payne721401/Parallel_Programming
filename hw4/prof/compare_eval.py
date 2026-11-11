"""Compare two evaluate.py output folders task by task.

A task is identical when it has the same success, the same metrics (timing fields
removed) and the same (level, horizon, num_samples, n_iters) for every planning call,
in order. The planner is deterministic, so any difference means some number changed.
Usage: compare_eval.py BASE_DIR NEW_DIR      (exit code 1 if anything differs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TIMING_WORDS = ("time", "wall", "clock", "second", "duration", "elapsed")


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def strip_timing(value):
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items()
                if not any(w in k.lower() for w in TIMING_WORDS)}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def plan_records(task):
    return [(r.get("level"), r.get("horizon"), r.get("num_samples"), tuple(r.get("n_iters") or ()))
            for r in task.get("planning_compute") or []]


def level_iters(records):
    total = {1: 0, 2: 0, 3: 0}
    for level, _, _, iters in records:
        total[int(level)] = total.get(int(level), 0) + sum(iters)
    return total


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base_dir, new_dir = (Path(p).expanduser() for p in sys.argv[1:])
    base = {p.name: load(p) for p in sorted((base_dir / "episodes").glob("task-*.json"))}
    new = {p.name: load(p) for p in sorted((new_dir / "episodes").glob("task-*.json"))}
    names = sorted(set(base) | set(new))
    rows, same_all = [], True
    for name in names:
        a, b = base.get(name), new.get(name)
        if not a or not b:
            rows.append([name, None, None, None, None, "missing in " + ("base" if not a else "new")] + [None] * 6)
            same_all = False
            continue
        ra, rb = plan_records(a), plan_records(b)
        diff_at = first_difference(ra, rb)
        same_metrics = strip_timing(a.get("metrics")) == strip_timing(b.get("metrics"))
        same_success = a.get("success") == b.get("success")
        same = same_success and same_metrics and diff_at is None and not b.get("error")
        same_all &= same
        ia, ib = level_iters(ra), level_iters(rb)
        note = "identical" if same else (b.get("error") or (f"plan call {diff_at} differs" if diff_at is not None
                                                            else "metrics differ" if not same_metrics else "success differs"))
        peak_a = (a.get("peak_allocated_bytes") or 0) / 2**20
        peak_b = (b.get("peak_allocated_bytes") or 0) / 2**20
        rows.append([name[5:7], b.get("success"), f"{ib[3]}/{ib[2]}/{ib[1]}",
                     "same" if (ia == ib) else f"base {ia[3]}/{ia[2]}/{ia[1]}", len(rb), note,
                     a.get("evaluation_wall_seconds"), b.get("evaluation_wall_seconds"),
                     a.get("worker_setup_seconds"), b.get("worker_setup_seconds"), peak_a, peak_b])
    print(f"base: {base_dir}\nnew:  {new_dir}")
    print(C.table(["task", "ok", "L3/L2/L1 it", "vs base", "plans", "result", "base wall s", "new wall s",
                   "base setup s", "new setup s", "base peak MiB", "new peak MiB"], rows, 1))
    for label, d in (("base", base_dir), ("new", new_dir)):
        report = load(d / "report.json") or {}
        print(f"{label}: wall {C.fmt(report.get('wall_seconds_this_invocation'))} s, "
              f"successes {report.get('successes')}/{report.get('episodes')}, "
              f"runtime errors {report.get('runtime_errors')}")
    print("IDENTICAL: every task matches the base" if same_all else "DIFFERENT: see the table above")
    return 0 if same_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
