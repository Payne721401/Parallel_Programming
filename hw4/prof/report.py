"""Summarize the outputs of the prof/ scripts. Pure Python, no GPU or torch needed.

    python prof/report.py ~/hw4-prof/a-<jobid> ~/hw4-prof/b-<jobid>
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

T = C.table


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def parse_time(path: Path):
    """bash `time` output -> {'real': s, 'user': s, 'sys': s}."""
    result = {}
    try:
        text = path.read_text()
    except OSError:
        return result
    for name, minutes, seconds in re.findall(r"^(real|user|sys)\s+(\d+)m([\d.]+)s", text, re.M):
        result[name] = int(minutes) * 60 + float(seconds)
    return result


def smi_stats(path: Path):
    util, power, mem = [], [], []

    def number(text):
        try:
            return float(text)
        except ValueError:
            return None

    try:
        with path.open() as stream:
            for row in csv.reader(stream):
                if len(row) < 5 or number(row[1]) is None:
                    continue
                util.append(number(row[1]))
                if number(row[3]) is not None:
                    mem.append(number(row[3]))
                if number(row[4]) is not None:
                    power.append(number(row[4]))
    except OSError:
        return None
    if not util:
        return None
    ordered = sorted(util)
    return {
        "samples": len(util),
        "util_mean": statistics.mean(util),
        "util_p50": ordered[len(ordered) // 2],
        "util_p90": ordered[int(len(ordered) * 0.9) - 1 if len(ordered) > 1 else 0],
        "frac_ge_90": sum(u >= 90 for u in util) / len(util),
        "power_mean_w": statistics.mean(power) if power else None,
        "mem_max_mib": max(mem) if mem else None,
    }


# ----------------------------------------------------------------------------- startup
def section_startup(d: Path) -> None:
    runs = sorted(d.glob("startup_*.json"))
    if not runs:
        return
    header("1. Startup breakdown of the check-case path (prof/startup.py)")
    reports = [load(p) for p in runs]
    reports = [r for r in reports if r]
    tags = [r["tag"] for r in reports]
    names = []
    for r in reports:
        for s in r["stages"]:
            if s["stage"] not in names:
                names.append(s["stage"])
    rows = [["interpreter start"] + [r.get("interpreter_start_s") for r in reports]]
    for name in names:
        row = [name]
        for r in reports:
            row.append(next((s["seconds"] for s in r["stages"] if s["stage"] == name), None))
        rows.append(row)
    rows.append(["= python side, check path"] + [r.get("python_total_s") for r in reports])
    walls, teardowns = [], []
    for r in reports:
        wall = teardown = None
        try:
            t0, t1 = (float(x) for x in (d / f"startup_wall_{r['tag']}.txt").read_text().split()[:2])
            wall = t1 - t0
            teardown = t1 - float(r["t_end_epoch"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        walls.append(wall)
        teardowns.append(teardown)
    rows.append(["interpreter exit (teardown)"] + teardowns)
    rows.append(["process wall incl. factory split"] + walls)
    print(T(["stage (s)"] + tags, rows, 3))
    last = reports[-1]
    check = last.get("check") or {}
    if check:
        print(f"\ncheck case {check.get('case')} (all: {check.get('all_case_ids')}), passed={check.get('passed')}, "
              f"{check.get('n_timed_calls')} timed calls/repeat, repeat wall ms "
              f"{[round(x, 1) for x in check.get('repeat_wall_ms', [])]}, cuda-event ms "
              f"{[round(x, 1) for x in check.get('repeat_cuda_event_ms', [])]}")
        print(f"pixels {check.get('pixels_shape')}, action banks {check.get('action_bank_shapes')}")
    split = last.get("factory_split")
    if split:
        print("\nbaseline build_model split (replica, run after the check path):")
        if "error" in split:
            print("  " + split["error"])
        else:
            print(T(["piece", "s"], [[k, v] for k, v in split.items()], 3))
    timing = parse_time(d / "check_real.time")
    real = load(d / "check_real.json")
    if timing or real:
        print("\nreal `check.py --case <first>` process:", ", ".join(f"{k} {v:.2f} s" for k, v in timing.items()))
        if real:
            print(f"  passed={real.get('passed')} median_ms={C.fmt(real.get('median_ms'))} "
                  f"cuda_event_median_ms={C.fmt(real.get('cuda_event_median_ms'))}")
    section_importtime(d / "importtime.txt")


def section_importtime(path: Path) -> None:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    entries = []
    for line in lines:
        m = re.match(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|(\s*)(\S+)", line)
        if m:
            depth = max(0, (len(m.group(3)) - 1) // 2)
            entries.append((int(m.group(2)) / 1e6, int(m.group(1)) / 1e6, depth, m.group(4)))
    if not entries:
        return
    top = sorted((e for e in entries if e[2] <= 1), key=lambda e: -e[0])[:14]
    print("\n`python -X importtime` (torch, transformers, ViT), top cumulative at depth <= 1:")
    print(T(["module", "cumulative s", "self s", "depth"], [[e[3], e[0], e[1], e[2]] for e in top], 3))


# -------------------------------------------------------------------------------- eval
def eval_tasks(folder: Path):
    tasks = []
    for path in sorted((folder / "episodes").glob("task-*.json")):
        r = load(path)
        if not r:
            continue
        per = defaultdict(lambda: [0, 0, 0.0])
        for rec in r.get("planning_compute") or []:
            entry = per[int(rec.get("level", 0))]
            entry[0] += 1
            entry[1] += sum(rec.get("n_iters") or [0])
            entry[2] += float(rec.get("wall_clock") or 0.0)
        tasks.append({"task": r.get("task_index"), "success": r.get("success"),
                      "completed": r.get("completed"), "wall": r.get("evaluation_wall_seconds"),
                      "setup": r.get("worker_setup_seconds"), "pid": (r.get("runtime") or {}).get("pid"),
                      "levels": per, "error": r.get("error")})
    return tasks


def section_eval(d: Path) -> dict:
    found = {}
    for name in ("eval8", "eval1"):
        folder = d / name
        report = load(folder / "report.json")
        if not report:
            continue
        tasks = eval_tasks(folder)
        found[name] = tasks
        header(f"2. evaluate.py {name}: workers={report.get('workers')} tasks={len(tasks)} "
               f"(--no-forward-timing, judge-like)")
        timing = parse_time(d / f"{name}.time")
        line = (f"wall this invocation {C.fmt(report.get('wall_seconds_this_invocation'))} s; "
                f"successes {report.get('successes')}/{report.get('episodes')}; runtime errors {report.get('runtime_errors')}")
        if timing.get("real"):
            busy = (timing.get("user", 0) + timing.get("sys", 0)) / timing["real"]
            line += f"; process real {timing['real']:.1f} s, user {timing.get('user', 0):.1f} s, sys {timing.get('sys', 0):.1f} s -> {busy:.2f} CPU cores busy on average"
        print(line)
        smi = smi_stats(d / f"smi_{name}.csv")
        if smi:
            print("nvidia-smi during run: " + ", ".join(f"{k}={C.fmt(v)}" for k, v in smi.items()))
        rows = []
        for t in tasks:
            lv = t["levels"]
            iters = {k: v[1] for k, v in lv.items()}
            solve = {k: v[2] for k, v in lv.items()}
            plan = sum(solve.values())
            rows.append([t["task"], t["success"], t["wall"], t["setup"], t["pid"], lv[1][0],
                         plan, solve.get(3), solve.get(2), solve.get(1), iters.get(3), iters.get(2), iters.get(1),
                         (solve.get(3, 0) / iters[3] * 1000) if iters.get(3) else None,
                         (solve.get(2, 0) / iters[2] * 1000) if iters.get(2) else None,
                         (solve.get(1, 0) / iters[1] * 1000) if iters.get(1) else None])
        print(T(["task", "ok", "wall s", "setup s", "pid", "plans", "solve s", "L3 s", "L2 s", "L1 s",
                 "L3 it", "L2 it", "L1 it", "L3 ms/it", "L2 ms/it", "L1 ms/it"], rows))
        by_pid = defaultdict(list)
        for t in tasks:
            by_pid[t["pid"]].append(t)
        rows = [[pid, [t["task"] for t in ts], ts[0]["setup"], sum(t["wall"] or 0 for t in ts),
                 (ts[0]["setup"] or 0) + sum(t["wall"] or 0 for t in ts)] for pid, ts in by_pid.items()]
        rows.sort(key=lambda r: -r[-1])
        print("per worker (critical path = setup + its tasks):")
        print(T(["pid", "tasks", "setup s", "tasks s", "setup+tasks s"], rows))
        for t in tasks:
            if t.get("error"):
                print(f"task {t['task']} error: {t['error']}")
    if "eval8" in found and "eval1" in found:
        e8 = {t["task"]: t for t in found["eval8"]}
        rows = []
        for t in found["eval1"]:
            other = e8.get(t["task"])
            if other and t["wall"] and other["wall"]:
                same = all(t["levels"][k][1] == other["levels"][k][1] for k in (1, 2, 3))
                rows.append([t["task"], t["wall"], other["wall"], other["wall"] / t["wall"], same])
        if rows:
            print("\ncontention: same task alone vs with 8 workers")
            print(T(["task", "1 worker s", "8 workers s", "slowdown x", "same iters"], rows))
    return found


# ------------------------------------------------------------------------ task profile
def section_task_profile(d: Path):
    data = load(d / "task_profile" / "task_profile.json")
    if not data:
        return None
    header("3. In-process task profile, 1 worker (prof/task_profile.py)")
    print(f"worker setup {C.fmt(data.get('worker_setup_s'))} s, {data.get('gpu')}, torch {data.get('torch')}")
    rows = []
    for t in data["tasks"]:
        a = t["acc"]
        env = a.get("world_step_s", 0) - a.get("get_action_s", 0)
        prep = a.get("get_action_s", 0) - a.get("plan_s", 0)
        plan_wo_enc = a.get("plan_s", 0) - a.get("encode_s", 0)
        rows.append([t["task"], t["success"], t["wall_s"], a.get("plan_s"), a.get("encode_s"), plan_wo_enc,
                     prep, env, t["wall_s"] - a.get("world_step_s", 0), int(a.get("plan_calls", 0)),
                     int(a.get("env_steps", 0))])
    print(T(["task", "ok", "wall s", "plan s", "encode s", "GD solves s", "obs prep s", "env step s",
             "outside steps s", "plans", "env steps"], rows))
    print("(encode = ViT+upper encoders for observation and goal, synchronized; obs prep = policy image "
          "transforms; outside steps = dataset/reset/world setup)")
    def order(item):
        match = re.match(r"L(\d+)_H(\d+)", item[0])
        return (-int(match.group(1)), -int(match.group(2))) if match else (0, 0)

    rows = []
    totals = defaultdict(float)
    for key, a in sorted(data["levels"].items(), key=order):
        iters = a["iters"] or 0
        if iters <= 0:
            continue
        per = lambda f: a[f] / iters * 1000  # noqa: E731
        loop = per("loop_s")
        other = loop - per("fwd_s") - per("bwd_s") - per("opt_s") - per("sync_s")
        non_loop = (a["init_s"] + a["clip_s"] + a["final_cost_s"] + a["final_rollout_s"]) / max(1, a["solves"]) * 1000
        rows.append([key, int(a["solves"]), int(a["iters_total"]), iters / max(1, a["solves"]), loop,
                     per("fwd_s"), per("rollout_s"), per("project_s"), per("criterion_s"),
                     per("bwd_s"), per("opt_s"), per("sync_s"), other, non_loop, a["solve_s"]])
        for f in ("solve_s", "loop_s"):
            totals[f] += a[f]
        for f in ("fwd_s", "bwd_s", "opt_s", "sync_s"):
            totals[f] += per(f) / 1000 * a["iters_total"]
    print("\nper (level, horizon), host ms per GD iteration (profiled iterations excluded):")
    print(T(["key", "solves", "iters", "it/solve", "loop", "fwd", " rollout", " project", " crit",
             "bwd", "opt", "sync", "other", "non-loop/solve", "solve s"], rows))
    if totals["solve_s"]:
        print(f"solve time {totals['solve_s']:.2f} s: fwd host {totals['fwd_s']:.2f}, bwd host {totals['bwd_s']:.2f}, "
              f"AdamW {totals['opt_s']:.2f}, sync wait {totals['sync_s']:.2f} s (estimates scaled to all iterations)")
    windows = data.get("windows") or []
    if windows:
        loop_ms = {}
        for key, a in data["levels"].items():
            if a["iters"]:
                loop_ms[key] = a["loop_s"] / a["iters"] * 1000
        rows = []
        for w in windows:
            busy = w.get("gpu_busy_ms_per_iter")
            gflop = w.get("gflop_per_iter")
            rows.append([w["key"], w.get("iters"), w.get("kernels_per_iter"), w.get("memops_per_iter"),
                         w.get("launch_calls_per_iter"), w.get("cpu_ops_per_iter"), busy, w.get("mean_kernel_us"),
                         gflop, (gflop / busy) if gflop and busy else None, loop_ms.get(w["key"]),
                         (busy / loop_ms[w["key"]]) if busy and loop_ms.get(w["key"]) else None])
        print("\ntorch.profiler windows (GPU kernels per GD iteration):")
        print(T(["key", "iters", "kernels", "memops", "launches", "cpu ops", "GPU busy ms", "mean kern us",
                 "GFLOP", "TFLOP/s", "loop ms (unprof.)", "GPU busy / loop"], rows, 3))
        for w in windows:
            cats = w.get("categories") or {}
            phases = w.get("phases") or {}
            print(f"\n[{w['key']}] kernel categories (count/iter, ms/iter): " + "; ".join(
                f"{k} {v['count_per_iter']:.0f}/{v['ms_per_iter']:.3f}" for k, v in list(cats.items())[:8]))
            print(f"[{w['key']}] by phase (count/iter, ms/iter): " + "; ".join(
                f"{k} {v['count_per_iter']:.0f}/{v['ms_per_iter']:.3f}" for k, v in phases.items()))
        first = windows[0]
        if first.get("top_kernels"):
            print(f"\n[{first['key']}] top kernels:")
            print(T(["kernel", "count/iter", "ms/iter"],
                    [[k["name"], k["count_per_iter"], k["ms_per_iter"]] for k in first["top_kernels"][:10]], 3))
        if first.get("cpu_table"):
            print(f"\n[{first['key']}] CPU ops by self CPU time (inflated by the profiler):")
            print(first["cpu_table"])
    return data


# -------------------------------------------------------------------------- microbench
def section_microbench(d: Path, task_data):
    single = load(d / "microbench.json")
    procs = sorted(d.glob("microbench_procs*.json"))
    if not single and not procs:
        return
    header("4. Microbench: replayed get_cost inputs, eager vs one CUDA graph (prof/microbench.py)")
    rows_by_name = {}
    if single:
        rows = []
        for r in single["results"]:
            p = r.get("profile") or {}
            rows_by_name[r["name"]] = r
            speed = (r["eager_full_ms"] / r["graph_full_ms"]) if r.get("eager_full_ms") and r.get("graph_full_ms") else None
            rows.append([r["name"], r["samples"], r.get("eager_fb_ms"), r.get("eager_full_ms"), r.get("host_fwd_ms"),
                         r.get("host_bwd_ms"), r.get("host_rest_ms"), r.get("sync_wait_ms"), p.get("kernels_per_iter"),
                         p.get("gpu_busy_ms_per_iter"), p.get("gflop_per_iter"), r.get("graph_fb_ms"),
                         r.get("graph_full_ms"), r.get("gpu_floor_ms"), speed,
                         r.get("graph_max_abs_cost"), r.get("graph_max_abs_grad")])
        print("single process, ms per GD iteration (median):")
        print(T(["case", "S", "eager fb", "eager full", "h fwd", "h bwd", "h rest", "sync", "kernels",
                 "GPU busy", "GFLOP", "graph fb", "graph full", "GPU floor", "full x", "|dcost|", "|dgrad|"],
                rows, 3))
        for r in single["results"]:
            for key in ("eager_error", "profile_error", "graph_error"):
                if r.get(key):
                    print(f"{r['name']} {key}:\n{r[key]}")
    for path in procs:
        data = load(path)
        if not data:
            continue
        print(f"\n{data['mode']}: {data['procs']} processes at once, files {data['files']}")
        rows = []
        for name, a in data["summary"].items():
            base = rows_by_name.get(name, {})
            rows.append([name, a.get("eager_full_ms_mean"), base.get("eager_full_ms"), a.get("sync_wait_ms_mean"),
                         a.get("graph_full_ms_mean"), base.get("graph_full_ms"), a.get("gpu_floor_ms_mean"),
                         base.get("gpu_floor_ms"), a.get("eager_aggregate_iters_per_s"), a.get("graph_aggregate_iters_per_s")])
        print(T(["case", "eager full", "(1 proc)", "sync", "graph full", "(1 proc)", "GPU floor", "(1 proc)",
                 "eager agg it/s", "graph agg it/s"], rows, 3))
        for name, a in data["summary"].items():
            if a.get("graph_error"):
                print(f"{name} graph capture failed:\n{a['graph_error']}")
        for err in data.get("errors") or []:
            print(err)
    if task_data and single:
        section_projection(task_data, single, procs)


def section_projection(task_data, single, procs):
    print("\nprojection onto the profiled task(s): GD-loop seconds = iterations x ms/iter")
    results = single["results"]
    contention = {}
    for path in procs:
        data = load(path)
        if data and data.get("procs") == 8:
            for name, a in data["summary"].items():
                base = next((r for r in results if r["name"] == name), None)
                if (base and base.get("eager_full_ms") and base.get("graph_full_ms")
                        and a.get("eager_full_ms_mean") and a.get("graph_full_ms_mean")):
                    level = name.split("_")[0]
                    contention[level] = (a["eager_full_ms_mean"] / base["eager_full_ms"],
                                         a["graph_full_ms_mean"] / base["graph_full_ms"])
    rows = []
    sums = defaultdict(float)
    for key, a in task_data["levels"].items():
        iters = a["iters_total"]
        if not iters:
            continue
        match = next((r for r in results if r["name"].startswith(key + "_")), None)
        if not match or not match.get("eager_full_ms"):
            rows.append([key, int(iters), a["loop_s"] / max(1, a["iters"]) * iters, None, None, None, None])
            continue
        level = key.split("_")[0]
        measured = a["loop_s"] / max(1, a["iters"]) * iters
        eager = iters * match["eager_full_ms"] / 1000
        graph = iters * match["graph_full_ms"] / 1000 if match.get("graph_full_ms") else None
        e8 = eager * contention[level][0] if level in contention else None
        g8 = graph * contention[level][1] if graph is not None and level in contention else None
        rows.append([key, int(iters), measured, eager, graph, e8, g8])
        for name, value in (("measured", measured), ("eager", eager), ("graph", graph), ("e8", e8), ("g8", g8)):
            if value is not None:
                sums[name] += value
    rows.append(["total (rows with data)", None, sums["measured"], sums["eager"], sums["graph"],
                 sums["e8"] or None, sums["g8"] or None])
    print(T(["key", "iters", "measured loop s", "eager est s", "graph est s", "eager@8 s", "graph@8 s"], rows))
    print("(@8 = scaled by the 8-process slowdown of the first case of that level)")


def main() -> int:
    dirs = [Path(p).expanduser() for p in sys.argv[1:]] or [C.out_dir()]
    task_data = None
    for d in dirs:
        if (d / "task_profile" / "task_profile.json").exists():
            task_data = load(d / "task_profile" / "task_profile.json")
    for d in dirs:
        print(f"\n##### {d}")
        section_startup(d)
        section_eval(d)
        data = section_task_profile(d)
        section_microbench(d, data or task_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
