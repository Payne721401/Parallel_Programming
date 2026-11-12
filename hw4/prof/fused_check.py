"""Check the Triton fused path (model/fused.py) against the current rollout on real inputs.

For every saved planner input: run the same GD loop as gd.py with HW4_FUSED off, then on
(both through the production CUDA graphs, so graph replay with changing inputs is covered).
  cost / grad rel   iteration 0: max |difference| / max |value|  (expect ~1e-6, FP32 rounding)
  cost@k rel        after the last iteration (differences grow with iterations; informational)
  off ms / on ms    median GPU-synchronised ms per GD iteration, single process
  capture s         first fused call: Triton JIT + graph capture
Exit code 1 if an iteration-0 difference exceeds --rtol. Writes $PROF_OUT/fused_check.json.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
from graph_check import clear_graphs, gd_loop, graphs_module  # noqa: E402
from microbench import Bench, input_files  # noqa: E402


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch

    bench = Bench()
    graphs = graphs_module(bench.model)
    jepa_module = sys.modules[type(bench.model.get_level(1)).__module__]
    fast, fused = jepa_module.fast_rollout, jepa_module.fast_rollout.fused
    levels = [bench.model.get_level(level) for level in range(1, bench.model.num_levels + 1)]

    def set_fused(on):
        fused.ENABLED = on
        for jepa in levels:
            fast.prepare(jepa)
            if jepa.__dict__.get(fast._FUSED, False) != on:
                raise RuntimeError(f"level {jepa.level}: fused path is {'not ' if on else ''}active; "
                                   "its structural requirements are not met")
        clear_graphs(bench.model, graphs)

    rows, results, ok = [], [], True
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        for path in input_files(args):
            case = bench.load_case(path)
            row = {"file": case.file}
            try:
                set_fused(False)
                gd_loop(bench, case, 3)
                off_costs, off_grad, _, off_times = gd_loop(bench, case, args.iters)
                set_fused(True)
                on_costs, on_grad, _, on_times = gd_loop(bench, case, args.iters)
                row.update(cost_rel=rel(on_costs[0], off_costs[0]), grad_rel=rel(on_grad, off_grad),
                           cost_last_rel=rel(on_costs[-1], off_costs[-1]),
                           off_ms=statistics.median(off_times) * 1000,
                           on_ms=statistics.median(on_times[1:]) * 1000,
                           capture_s=on_times[0] - statistics.median(on_times[1:]))
                finite = all(bool(torch.isfinite(c).all()) for c in on_costs) and bool(torch.isfinite(on_grad).all())
                ok &= finite and row["cost_rel"] <= args.rtol and row["grad_rel"] <= args.rtol
                row["finite"] = finite
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {str(error)[:300]}"
                ok = False
            results.append(row)
            rows.append([case.file, row.get("cost_rel"), row.get("grad_rel"), row.get("cost_last_rel"),
                         row.get("off_ms"), row.get("on_ms"),
                         (row["off_ms"] / row["on_ms"]) if row.get("on_ms") else None,
                         row.get("capture_s"), row.get("error", "")])
            print(f"[{case.file}] " + ", ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
                                                  for k, v in row.items() if k != "file"), flush=True)
    set_fused(False)
    print("\nfused (Triton) vs current rollout, same GD loop:")
    print(C.table(["input", "cost rel", "grad rel", f"cost@{args.iters} rel", "off ms", "on ms", "x",
                   "capture s", "error"], [[r[0]] + [None if v is None else (f"{v:.2e}" if i < 3 else v)
                                                      for i, v in enumerate(r[1:-1])] + [r[-1]] for r in rows], 3))
    print("PASS: iteration-0 cost and gradient agree within rtol" if ok else "FAIL: see the table above")
    out = C.out_dir()
    C.save_json(out / "fused_check.json", {"iters": args.iters, "rtol": args.rtol, "passed": ok, "results": results})
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
