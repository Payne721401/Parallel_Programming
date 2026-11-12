"""Compare each attention candidate against off on saved real planner inputs.

Both sides use production rollout CUDA graphs. Check every cost, action gradient
and post-update action over a fixed 90-step stress solve (not a timed evaluation).
The absolute tolerance is a diagnostic gate, not an official gradient tolerance.
Run check.py and full tasks separately; never infer success rate from this probe.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from graph_check import clear_graphs, graphs_module
from microbench import Bench, input_files


def solve(bench, case, iters):
    torch = bench.torch
    p = case.actions.detach().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([p], lr=case.lr)
    history = {"cost": [], "gradient": [], "actions": []}
    for _ in range(iters):
        cost = case.cm.get_cost(dict(case.info), p)
        cost.sum().backward()
        history["cost"].append(cost.detach().cpu().clone())
        history["gradient"].append(p.grad.detach().cpu().clone())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if case.bounds is not None:
            with torch.no_grad():
                p.clamp_(min=case.bounds[0], max=case.bounds[1])
        history["actions"].append(p.detach().cpu().clone())
        cost.min(dim=1).values.detach().cpu().tolist()
    # Exercise the eager/no-grad final-cost path too.
    with torch.no_grad():
        history["final_cost"] = [case.cm.get_cost(dict(case.info), p).detach().cpu().clone()]
    return {key: torch.stack(values) for key, values in history.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--modes", nargs="+", choices=("sdpa", "v"), default=["sdpa", "v"])
    parser.add_argument("--iters", type=int, default=90)
    parser.add_argument("--atol", type=float, default=3e-4)
    args = parser.parse_args()
    if args.iters < 1 or not (0 < args.atol < float("inf")):
        parser.error("iters and finite atol must be positive")
    files = input_files(args)  # fail closed on an empty input folder

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bench = Bench()
    graphs = graphs_module(bench.model)
    attention = next(m for m in bench.model.modules() if type(m).__name__ == "Attention")
    modules = sys.modules[type(attention).__module__]
    original_mode, original_graphs = modules.SINGLE_TOKEN_ATTENTION, graphs.ENABLED
    rows = []
    report = {"passed": False, "iters": args.iters, "atol": args.atol,
              "reference": "off + production rollout graph", "results": rows,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    out = C.out_dir() / "attention_planner_check.json"
    C.save_json(out, report)
    try:
        graphs.ENABLED = True
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            for path in files:
                case = bench.load_case(path)
                reference = None
                for mode in ("off", *args.modes):
                    # A CUDA graph bakes in the branch. Never reuse a graph
                    # captured in another mode when switching this diagnostic.
                    clear_graphs(bench.model, graphs)
                    modules.SINGLE_TOKEN_ATTENTION = mode
                    actual = solve(bench, case, args.iters)
                    caches = [entry for level in range(1, bench.model.num_levels + 1)
                              for entry in bench.model.get_level(level).__dict__.get(graphs._CACHE_ATTR, {}).values()]
                    captured = bool(caches) and all(callable(entry) for entry in caches)
                    if not captured:
                        raise RuntimeError(f"{path.name}/{mode}: production graph capture missing or failed")
                    if not all(bool(torch.isfinite(value).all()) for value in actual.values()):
                        raise RuntimeError(f"{path.name}/{mode}: non-finite solve values")
                    if mode == "off":
                        reference = actual
                        continue
                    diffs = {key: float((actual[key] - reference[key]).abs().max()) for key in actual}
                    row = {"file": path.name, "mode": mode, "max_abs": diffs,
                           "passed": all(value <= args.atol for value in diffs.values()),
                           "captured_graphs": len(caches)}
                    rows.append(row)
                    print(row, flush=True)
                    C.save_json(out, report)
        report["passed"] = bool(rows) and all(row["passed"] for row in rows)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        clear_graphs(bench.model, graphs)
        modules.SINGLE_TOKEN_ATTENTION, graphs.ENABLED = original_mode, original_graphs
        C.save_json(out, report)
        print(f"planner check: passed={report['passed']}; {out}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
