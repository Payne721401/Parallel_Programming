"""Check the rollout CUDA graphs (model/graphs.py) against eager on real planner inputs.

For every get_cost input saved by task_profile.py (L3 H6..H2, L2, L1 in each goal space),
run the same GD loop as gd.py (get_cost, backward, AdamW, clamp, min().cpu()) twice
from the same start: once with HW4_ROLLOUT_GRAPH off, once with the graphs on.
  cost diff     max |cost| difference over every iteration      (must be exactly 0)
  grad diff     max |action gradient| difference, iteration 0   (must be exactly 0)
  action diff   max |action| difference after the last iteration (must be exactly 0)
  capture s     first graphed call (warm-up + capture + replay) minus one replay
  +MiB          memory still allocated after capture (static buffers and graph pool)
  eager/graph   median ms per GD iteration in this single process
Writes $PROF_OUT/graph_check.json; exits 1 if any difference is not exactly 0.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
from microbench import Bench, input_files  # noqa: E402

perf = time.perf_counter


def graphs_module(model):
    jepa_module = sys.modules[type(model.get_level(1)).__module__]
    return sys.modules[jepa_module.rollout_graph.__module__]


def clear_graphs(model, graphs):
    for level in range(1, model.num_levels + 1):
        model.get_level(level).__dict__.pop(graphs._CACHE_ATTR, None)


def gd_loop(bench, case, iters):
    """One solve's worth of GD iterations, as in gd.py (batch_size 1, no action noise)."""
    torch = bench.torch
    p = case.actions.detach().clone().requires_grad_(True)
    opt = torch.optim.AdamW([p], lr=case.lr)
    costs, times, grad0 = [], [], None
    for i in range(iters):
        torch.cuda.synchronize()
        t = perf()
        cost = case.cm.get_cost(dict(case.info), p)
        cost.sum().backward()
        if i == 0:
            grad0 = p.grad.detach().clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if case.bounds is not None:
            p.data.clamp_(min=case.bounds[0], max=case.bounds[1])
        cost.min(dim=1).values.detach().cpu().tolist()
        times.append(perf() - t)
        costs.append(cost.detach().clone())
    return costs, grad0, p.detach().clone(), times


def max_abs(a, b):
    return float((a - b).abs().max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=None, help="folder of captured *.pt (default from PROF_OUT)")
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch

    bench = Bench()
    graphs = graphs_module(bench.model)
    rows, results, exact = [], [], True
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        for path in input_files(args):
            case = bench.load_case(path)
            row = {"file": case.file, "level": case.level, "horizon": case.horizon, "samples": case.samples}
            try:
                graphs.ENABLED = False
                gd_loop(bench, case, 3)  # warm the eager path
                e_costs, e_grad, e_actions, e_times = gd_loop(bench, case, args.iters)

                graphs.ENABLED = True
                clear_graphs(bench.model, graphs)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                g_costs, g_grad, g_actions, g_times = gd_loop(bench, case, args.iters)
                after = torch.cuda.memory_allocated()

                row["cost_diff"] = max(max_abs(a, b) for a, b in zip(e_costs, g_costs))
                row["grad_diff"] = max_abs(e_grad, g_grad)
                row["action_diff"] = max_abs(e_actions, g_actions)
                row["eager_ms"] = statistics.median(e_times) * 1000
                row["graph_ms"] = statistics.median(g_times[1:]) * 1000
                row["capture_s"] = g_times[0] - statistics.median(g_times[1:])
                row["graph_mib"] = (after - before) / 2**20
                row["cached_graphs"] = sum(len(bench.model.get_level(l).__dict__.get(graphs._CACHE_ATTR, {}))
                                           for l in range(1, bench.model.num_levels + 1))
                exact &= row["cost_diff"] == 0 and row["grad_diff"] == 0 and row["action_diff"] == 0
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                exact = False
            finally:
                graphs.ENABLED = True
                clear_graphs(bench.model, graphs)
                torch.cuda.empty_cache()
            results.append(row)
            rows.append([case.file, row.get("cost_diff"), row.get("grad_diff"), row.get("action_diff"),
                         row.get("eager_ms"), row.get("graph_ms"),
                         (row["eager_ms"] / row["graph_ms"]) if row.get("graph_ms") else None,
                         row.get("capture_s"), row.get("graph_mib"), row.get("error", "")])
            print(f"[{case.file}] " + ", ".join(f"{k}={C.fmt(v, 4)}" for k, v in row.items()
                                                  if k not in ("file",)), flush=True)

    print("\nrollout graph vs eager, same GD loop (diffs must be exactly 0):")
    print(C.table(["input", "cost diff", "grad diff", "action diff", "eager ms", "graph ms", "x",
                   "capture s", "+MiB", "error"], rows, 3))
    print("EXACT: all differences are 0" if exact else "NOT EXACT: see the table above")
    out = C.out_dir()
    C.save_json(out / "graph_check.json", {"iters": args.iters, "exact": exact, "results": results,
                                           "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
    print(f"graph_check -> {out / 'graph_check.json'}")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
