"""Nano5 attention gates and fresh-process A/B/B/A e1 timing in one allocation.

All output is outside hw4. No default is promoted automatically. A metrics
difference is reported separately from invalid/incomplete execution; even a
successful e1 comparison still needs the full 50-task success-rate check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _common as C


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_check(path):
    report = read_json(path)
    cases = report.get("cases") or []
    require(report.get("completed") is True and report.get("passed") is True
            and not report.get("error") and report.get("epsilon") == 3e-4,
            f"check did not pass: {path}")
    require(len(cases) == 6 and len({row.get('id') for row in cases}) == 6
            and all(row.get("passed") is True for row in cases),
            f"expected all six check cases: {path}")


def validate_eval(folder):
    report = read_json(folder / "report.json")
    require(report.get("finished") is True and report.get("completed") is True
            and report.get("runtime_errors") == 0 and not report.get("error")
            and report.get("episodes") == report.get("completed_episodes") == 10
            and report.get("workers") == 8
            and report.get("full_run_timing_available") is True,
            f"invalid/incomplete e1 report: {folder}")
    files = sorted((folder / "episodes").glob("task-*.json"))
    require({p.name for p in files} == {f"task-{i:02d}.json" for i in range(10)},
            f"missing or unexpected e1 tasks: {folder}")
    successes = 0
    for i, path in enumerate(files):
        row = read_json(path)
        seed = row.get("environment_seed")
        require(row.get("task_index") == i and row.get("planner_seed") == 42 + i
                and row.get("completed") is True and not row.get("error")
                and row.get("environment_seed_verified") is True
                and isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
                and row.get("environment_seed_observed") == seed
                and row.get("metrics", {}).get("seeds") == [seed]
                and row.get("environment_reset_audit", {}).get("applied") is True
                and row.get("environment_reset_audit", {}).get("python_integer_seeds") == [seed],
                f"task/seed/error validation failed: {path}")
        require(type(row.get("success")) is bool
                and row.get("metrics", {}).get("episode_successes") == [row["success"]],
                f"inconsistent success: {path}")
        successes += row["success"]
    require(report.get("successes") == successes, f"inconsistent success total: {folder}")
    return report


def run(command, env, log, *, comparison=False):
    print(f"start: {' '.join(map(str, command))}\nlog: {log}", flush=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run([str(x) for x in command], cwd=C.ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - started
    require(process.returncode in ((0, 1) if comparison else (0,)),
            f"exit {process.returncode}; inspect {log}")
    print(f"finished: exit={process.returncode}, process wall={seconds:.3f} s", flush=True)
    return seconds, process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sdpa", "v"), required=True)
    parser.add_argument("--baseline", choices=("off", "sdpa"), default="off")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.baseline != args.mode, "candidate and baseline must differ")
    require(args.inputs.is_dir() and any(args.inputs.glob("*.pt")), "no captured planner inputs")
    out = args.output.expanduser().resolve()
    require(not out.is_relative_to(C.ROOT.resolve()), "output must be outside hw4")
    out.mkdir(parents=True, exist_ok=False)  # no stale or resumed results
    env = dict(os.environ, HW4_ROLLOUT_GRAPH="1", HW4_GRAPH_LOG="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PROF_OUT=str(out))
    report = {"completed": False, "baseline": args.baseline, "candidate": args.mode,
              "order": [args.baseline, args.mode, args.mode, args.baseline],
              "needs_full_50_task_validation": True, "runs": [], "source_sha256": {},
              "environment": {k: v for k, v in env.items() if k.startswith("HW4_") or
                              k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
    for source in [C.ROOT / "hw4.py", *sorted((C.ROOT / "model").glob("*.py"))]:
        relative = source.relative_to(C.ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        report["source_sha256"][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = out / "summary.json"
    C.save_json(summary, report)
    try:
        run([sys.executable, "prof/attention_check.py", "--device", "cuda",
             "--output", out / "attention_check.json"], env, out / "attention_check.log")
        for mode in (args.baseline, args.mode):
            mode_env = dict(env, HW4_SINGLE_TOKEN_ATTN=mode)
            path = out / f"check-{mode}.json"
            run([sys.executable, "check.py", "--output", path], mode_env, out / f"check-{mode}.log")
            validate_check(path)  # check.py also returns 0 for a numerical failure
        run([sys.executable, "prof/attention_planner_check.py", "--inputs", args.inputs.resolve(),
             "--modes", args.mode, "--iters", "90"], env, out / "planner_check.log")
        for index, mode in enumerate(report["order"]):
            folder = out / f"eval-{index + 1}-{mode}"
            seconds, _ = run([sys.executable, "evaluate.py", "--workers", "8", "--tasks",
                              *range(10), "--no-forward-timing", "--output-dir", folder],
                             dict(env, HW4_SINGLE_TOKEN_ATTN=mode), out / f"eval-{index + 1}-{mode}.log")
            evaluation = validate_eval(folder)
            row = {"mode": mode, "directory": str(folder), "process_wall_s": seconds,
                   "evaluation_wall_s": evaluation["wall_seconds_this_invocation"],
                   "successes": evaluation["successes"]}
            report["runs"].append(row)
            print(row, flush=True)
            C.save_json(summary, report)
        for index in (1, 2, 3):
            _, code = run([sys.executable, "prof/compare_eval.py", report["runs"][0]["directory"],
                           report["runs"][index]["directory"]], env, out / f"compare-1-{index + 1}.log",
                          comparison=True)
            report["runs"][index]["recorded_metrics_match_first_baseline"] = code == 0
        base = [r for r in report["runs"] if r["mode"] == args.baseline]
        candidate = [r for r in report["runs"] if r["mode"] == args.mode]
        base_mean = sum(r["process_wall_s"] for r in base) / len(base)
        new_mean = sum(r["process_wall_s"] for r in candidate) / len(candidate)
        report.update(completed=True, baseline_mean_s=base_mean, candidate_mean_s=new_mean,
                      speedup=base_mean / new_mean,
                      e1_successes_preserved=min(r["successes"] for r in candidate) >=
                                             max(r["successes"] for r in base))
        print(f"A/B/B/A complete: {base_mean:.3f} -> {new_mean:.3f} s, {base_mean / new_mean:.4f}x\n"
              f"e1 successes preserved: {report['e1_successes_preserved']}; full 50 tasks still required.", flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        C.save_json(summary, report)
        print(f"summary: {summary}", flush=True)
    return 0 if report["e1_successes_preserved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
