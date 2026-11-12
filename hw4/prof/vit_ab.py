"""Verify native ViT and measure fresh check/e1 processes in one GPU allocation.

Keep attention mode fixed while changing only HW4_VIT_IMPL. All outputs, source
snapshots, logs and timing reports are written outside hw4. Does not pre-import
transformers in any timed child or require identical optimization trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

import _common as C
from attention_ab import read_json, require, run, validate_check, validate_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention", choices=("off", "sdpa", "v"), default="v")
    parser.add_argument("--checks-only", action="store_true", help="skip the four e1 runs")
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    require(not out.is_relative_to(C.ROOT.resolve()), "output must be outside hw4")
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HW4_SINGLE_TOKEN_ATTN=args.attention,
               HW4_ROLLOUT_GRAPH="1", HW4_GRAPH_LOG="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    report = {"completed": False, "passed": False, "attention": args.attention,
              "order": ["hf", "native", "native", "hf"], "check_runs": [], "eval_runs": [],
              "source_sha256": {}, "environment": {k: v for k, v in env.items()
                    if k.startswith("HW4_") or k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
    summary = out / "summary.json"
    for source in [C.ROOT / "hw4.py", *sorted((C.ROOT / "model").glob("*.py"))]:
        relative = source.relative_to(C.ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        report["source_sha256"][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    C.save_json(summary, report)
    try:
        # A fresh native factory must build the released model without importing HF.
        probe = ("import sys,runner; m,_,_=runner.load_model(); "
                 "assert 'transformers' not in sys.modules, 'native build imported transformers'; "
                 "print(type(m.get_level(1).encoder.encoder).__name__); "
                 "print('native factory: no transformers import')")
        run([sys.executable, "-c", probe], dict(env, HW4_VIT_IMPL="native"), out / "native-import.log")
        run([sys.executable, "prof/vit_check.py", "--device", "cuda", "--assets",
             "--output", out / "vit_check.json"], env, out / "vit_check.log")
        for mode in ("hf", "native"):
            path = out / f"check-all-{mode}.json"
            run([sys.executable, "check.py", "--output", path],
                dict(env, HW4_VIT_IMPL=mode), out / f"check-all-{mode}.log")
            validate_check(path)
        for index, mode in enumerate(report["order"], 1):
            path = out / f"check-01-{index}-{mode}.json"
            seconds, _ = run([sys.executable, "check.py", "--case", "01", "--output", path],
                             dict(env, HW4_VIT_IMPL=mode), out / f"check-01-{index}-{mode}.log")
            check = read_json(path)
            require(check.get("completed") is True and check.get("passed") is True
                    and not check.get("error") and len(check.get("cases", [])) == 1
                    and check["cases"][0].get("id") == "01", f"check 01 failed: {path}")
            report["check_runs"].append({"mode": mode, "process_wall_s": seconds,
                                         "max_abs": check["cases"][0]["max_abs"]})
            C.save_json(summary, report)
        if not args.checks_only:
            for index, mode in enumerate(report["order"], 1):
                folder = out / f"eval-{index}-{mode}"
                seconds, _ = run([sys.executable, "evaluate.py", "--workers", "8", "--tasks",
                                  *range(10), "--no-forward-timing", "--output-dir", folder],
                                 dict(env, HW4_VIT_IMPL=mode), out / f"eval-{index}-{mode}.log")
                evaluation = validate_eval(folder)
                row = {"mode": mode, "directory": str(folder), "process_wall_s": seconds,
                       "evaluation_wall_s": evaluation["wall_seconds_this_invocation"],
                       "successes": evaluation["successes"]}
                report["eval_runs"].append(row)
                C.save_json(summary, report)
                print(row, flush=True)
            for index in (1, 2, 3):
                _, code = run([sys.executable, "prof/compare_eval.py", report["eval_runs"][0]["directory"],
                               report["eval_runs"][index]["directory"]], env,
                              out / f"compare-1-{index + 1}.log", comparison=True)
                report["eval_runs"][index]["recorded_metrics_match_first_baseline"] = code == 0
            report["e1_successes_preserved"] = (
                min(r["successes"] for r in report["eval_runs"] if r["mode"] == "native") >=
                max(r["successes"] for r in report["eval_runs"] if r["mode"] == "hf"))
        for section in ("check_runs", "eval_runs"):
            rows = report[section]
            if not rows:
                continue
            means = {mode: sum(r["process_wall_s"] for r in rows if r["mode"] == mode) / 2
                     for mode in ("hf", "native")}
            report[section + "_summary"] = dict(means, saved_s=means["hf"] - means["native"],
                                                 speedup=means["hf"] / means["native"])
            print(f"{section}: {report[section + '_summary']}", flush=True)
        report["completed"] = True
        report["passed"] = report.get("e1_successes_preserved", True)
        report["needs_full_50_task_confirmation"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        C.save_json(summary, report)
        print(f"summary: {summary}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
