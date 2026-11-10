"""In-process profile of FourRoom planning tasks with one worker.

Runs evaluate.py's own _worker_init/_evaluate_task (same threads, SDPA MATH, seeds), with
timing wrappers around the fixed planner and the model. Planner behaviour is unchanged.

Per (level, horizon) it reports GD iterations and the per-iteration HOST time of
  fwd  = LevelCostModel.get_cost (rollout + projection up + criterion), launched async
  bwd  = cost.backward()
  opt  = AdamW.step
  sync = the early-stop .cpu() in gd.py; its wait is how far the GPU lags the CPU
  other= the rest of the loop (zero_grad, clamp, min, Python)
For one planning call (--trace-call) it records a torch.profiler window of --trace-iters
iterations per level (kernels/iter, GPU busy time, kernel categories, FLOPs); those
iterations are excluded from the host-time averages. It also saves the first get_cost
inputs of every (level, horizon, goal space) for prof/microbench.py.
Writes $PROF_OUT/task_profile/task_profile.json, traces and inputs/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.setup_paths()
os.environ.setdefault("HW4_ASSETS_VERIFIED", "1")

perf = time.perf_counter
LEVEL_FIELDS = ("solves", "solve_s", "iters", "iters_total", "loop_s", "fwd_s", "rollout_s",
                "project_s", "criterion_s", "bwd_s", "opt_s", "sync_s", "sync_n", "init_s",
                "clip_s", "final_cost_s", "final_rollout_s")


class Instrument:
    def __init__(self, args, out):
        import torch

        self.torch = torch
        self.args = args
        self.out = out
        self.task = None
        self.first_task = None
        self.plan_call = 0
        self.level = None
        self.horizon = None
        self.iter = 0
        self.loop_active = False
        self.loop_t0 = 0.0
        self.excluded_s = 0.0
        self.excluded_iters = 0
        self.capturing = False
        self.profiling = False
        self.prof = None
        self.prof_iters = 0
        self.prof_t_active = 0.0
        self.ws_t0 = 0.0
        self.traced_levels = set()
        self.captured = {}
        self.capture_count = 0
        self.levels = defaultdict(lambda: defaultdict(float))
        self.task_acc = defaultdict(float)
        self.tasks = []
        self.windows = []
        self.calls = []

    # ------------------------------------------------------------------ helpers
    def key(self):
        return f"L{self.level}_H{self.horizon}"

    def acc(self):
        return self.levels[self.key()]

    def ann(self, name):
        if self.profiling:
            return self.torch.profiler.record_function("prof:" + name)
        return nullcontext()

    def counting(self):
        return self.loop_active and self.level is not None and not self.capturing

    # ------------------------------------------------------------- GD loop marks
    def close_loop(self):
        acc = self.acc()
        acc["loop_s"] += perf() - self.loop_t0 - self.excluded_s
        acc["iters"] += self.iter - self.excluded_iters
        acc["iters_total"] += self.iter
        self.loop_active = False

    def maybe_toggle_profiler(self):
        a = self.args
        if a.trace_call <= 0 or self.task != self.first_task or self.plan_call != a.trace_call:
            return
        start_it = a.trace_skip + 1
        if not self.profiling and self.iter == start_it and self.level not in self.traced_levels:
            self.start_profiler()
        elif self.profiling and self.iter == start_it + a.trace_iters:
            self.stop_profiler()

    def start_profiler(self):
        from torch.profiler import ProfilerActivity, profile

        self.ws_t0 = perf()
        self.torch.cuda.synchronize()
        self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=True, with_flops=True)
        self.prof.__enter__()
        self.profiling = True
        self.prof_iters = 0
        self.traced_levels.add(self.level)
        self.prof_t_active = perf()

    def stop_profiler(self):
        torch = self.torch
        torch.cuda.synchronize()
        active = perf() - self.prof_t_active
        self.prof.__exit__(None, None, None)
        self.profiling = False
        name = self.key()
        info = {"key": name, "plan_call": self.plan_call, "iters": self.prof_iters,
                "first_iter": self.args.trace_skip + 1,
                "wall_ms_per_iter_under_profiler": active * 1000 / max(1, self.prof_iters)}
        try:
            trace = self.out / f"trace_{name}.json"
            self.prof.export_chrome_trace(str(trace))
            info.update(C.analyze_trace(trace, self.prof_iters))
            info["trace"] = str(trace)
        except Exception as error:
            info["trace_error"] = repr(error)
        try:
            averages = self.prof.key_averages()
            flops = sum(float(getattr(e, "flops", 0) or 0) for e in averages)
            info["gflop_per_iter"] = flops / 1e9 / max(1, self.prof_iters)
            info["cpu_table"] = averages.table(sort_by="self_cpu_time_total", row_limit=15)
        except Exception as error:
            info["key_averages_error"] = repr(error)
        self.windows.append(info)
        self.prof = None
        self.excluded_s += perf() - self.ws_t0
        self.excluded_iters += self.prof_iters
        print(f"  profiled {name} (plan call {self.plan_call}): "
              f"{info.get('kernels_per_iter', float('nan')):.0f} kernels/iter, "
              f"GPU busy {info.get('gpu_busy_ms_per_iter', float('nan')):.3f} ms/iter", flush=True)

    def maybe_capture(self, info, actions):
        torch = self.torch
        dense = bool(info.get("dense_upper_goal_projection", False))
        upper = bool(info.get("goal_embed_0_in_upper_space", True))
        goal_level = info.get("goal_embed_0_level")
        space = "dense" if dense else ("up" if upper else "own")
        name = f"L{self.level}_H{self.horizon}_{space}" + (f"{goal_level}" if goal_level else "")
        if name in self.captured:
            return
        t = perf()
        self.capturing = True
        try:
            samples = actions.shape[1]
            # Only tensors and plain scalars: the model path reads nothing else, and it
            # keeps the file loadable with torch.load(weights_only=True).
            tensors, meta, dropped = {}, {}, []
            for key, value in info.items():
                if torch.is_tensor(value):
                    d = value.detach()
                    if d.ndim >= 2 and d.shape[1] == samples and d.stride(1) == 0:
                        tensors[key] = ["expand1", d[:, :1].cpu().clone()]
                    else:
                        tensors[key] = ["plain", d.cpu().clone()]
                elif isinstance(value, (bool, int, float, str, type(None))):
                    meta[key] = value
                else:
                    dropped.append(f"{key}:{type(value).__name__}")
            blob = {"name": name, "level": self.level, "horizon": self.horizon,
                    "num_samples": samples, "tensors": tensors, "meta": meta,
                    "dropped": dropped, "actions": actions.detach().cpu().clone(),
                    "task": self.task, "plan_call": self.plan_call}
            self.capture_count += 1
            path = self.out / "inputs" / f"{self.capture_count:02d}_{name}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(blob, path)
            self.captured[name] = str(path)
        except Exception as error:
            self.captured[name] = f"error: {error!r}"
        finally:
            self.capturing = False
            self.excluded_s += perf() - t

    # ---------------------------------------------------------------- install
    def install(self, model):
        torch = self.torch
        import stable_worldmodel  # noqa: F401
        from stable_worldmodel.policy import WorldModelPolicy
        from stable_worldmodel.solver.gd import GradientSolver
        from stable_worldmodel.world import World
        from hierarchical_solver import HierarchicalSolver, LevelCostModel

        inst = self

        def patch(owner, name, make):
            setattr(owner, name, make(getattr(owner, name)))

        def timed_task(field, count_field=None):
            def make(original):
                def wrapper(self_, *a, **k):
                    t = perf()
                    try:
                        return original(self_, *a, **k)
                    finally:
                        inst.task_acc[field] += perf() - t
                        if count_field:
                            inst.task_acc[count_field] += 1
                return wrapper
            return make

        patch(World, "step", timed_task("world_step_s", "env_steps"))
        patch(WorldModelPolicy, "get_action", timed_task("get_action_s"))

        def h_solve(original):
            def wrapper(self_, *a, **k):
                inst.plan_call += 1
                t = perf()
                try:
                    return original(self_, *a, **k)
                finally:
                    dt = perf() - t
                    inst.task_acc["plan_s"] += dt
                    inst.task_acc["plan_calls"] += 1
                    inst.calls.append({"task": inst.task, "call": inst.plan_call,
                                       "steps_taken": k.get("steps_taken"), "seconds": dt})
            return wrapper

        patch(HierarchicalSolver, "solve", h_solve)

        def encode_all(original):
            def wrapper(self_, *a, **k):
                torch.cuda.synchronize()
                t = perf()
                try:
                    return original(self_, *a, **k)
                finally:
                    torch.cuda.synchronize()
                    inst.task_acc["encode_s"] += perf() - t
            return wrapper

        patch(HierarchicalSolver, "_encode_all_levels", encode_all)

        def gd_solve(original):
            def wrapper(self_, *a, **k):
                inst.level = int(getattr(self_, "level", 0))
                horizon = k.get("planning_horizon")
                if horizon is None and len(a) >= 3:
                    horizon = a[2]
                inst.horizon = horizon
                inst.iter = 0
                inst.loop_active = False
                inst.excluded_s = 0.0
                inst.excluded_iters = 0
                t = perf()
                try:
                    return original(self_, *a, **k)
                finally:
                    if inst.profiling:
                        inst.stop_profiler()
                    if inst.loop_active:
                        inst.close_loop()
                    acc = inst.acc()
                    acc["solves"] += 1
                    acc["solve_s"] += perf() - t
                    inst.level = None
            return wrapper

        patch(GradientSolver, "solve", gd_solve)

        def level_timed(field):
            def make(original):
                def wrapper(self_, *a, **k):
                    t = perf()
                    try:
                        return original(self_, *a, **k)
                    finally:
                        if inst.level is not None:
                            inst.acc()[field] += perf() - t
                return wrapper
            return make

        patch(GradientSolver, "init_action", level_timed("init_s"))
        patch(GradientSolver, "_get_action_clip_bounds", level_timed("clip_s"))
        patch(GradientSolver, "_rollout_final_actions", level_timed("final_rollout_s"))

        def get_cost(original):
            def wrapper(self_, info_dict, action_candidates):
                if not torch.is_grad_enabled():  # the final cost after the loop
                    if inst.profiling:
                        inst.stop_profiler()
                    if inst.loop_active:
                        inst.close_loop()
                    t = perf()
                    try:
                        return original(self_, info_dict, action_candidates)
                    finally:
                        if inst.level is not None:
                            inst.acc()["final_cost_s"] += perf() - t
                inst.iter += 1
                if inst.iter == 1:
                    inst.loop_active = True
                    inst.loop_t0 = perf()
                    if inst.args.capture:
                        inst.maybe_capture(info_dict, action_candidates)
                inst.maybe_toggle_profiler()
                if inst.profiling:
                    inst.prof_iters += 1
                t = perf()
                with inst.ann("fwd"):
                    result = original(self_, info_dict, action_candidates)
                if not inst.profiling:
                    inst.acc()["fwd_s"] += perf() - t
                return result
            return wrapper

        patch(LevelCostModel, "get_cost", get_cost)

        def in_loop_timed(field):
            def make(original):
                def wrapper(self_, *a, **k):
                    if not (inst.counting() and torch.is_grad_enabled()) or inst.profiling:
                        return original(self_, *a, **k)
                    t = perf()
                    try:
                        return original(self_, *a, **k)
                    finally:
                        inst.acc()[field] += perf() - t
                return wrapper
            return make

        patch(LevelCostModel, "rollout", in_loop_timed("rollout_s"))
        patch(LevelCostModel, "_project_to_goal_space", in_loop_timed("project_s"))
        jepa_cls = type(model.get_level(1))
        if hasattr(jepa_cls, "criterion"):
            patch(jepa_cls, "criterion", in_loop_timed("criterion_s"))

        def phase_timed(field, phase, count_field=None):
            def make(original):
                def wrapper(self_, *a, **k):
                    if not inst.counting():
                        return original(self_, *a, **k)
                    t = perf()
                    with inst.ann(phase):
                        result = original(self_, *a, **k)
                    if not inst.profiling:
                        acc = inst.acc()
                        acc[field] += perf() - t
                        if count_field:
                            acc[count_field] += 1
                    return result
                return wrapper
            return make

        patch(torch.Tensor, "backward", phase_timed("bwd_s", "bwd"))
        patch(torch.optim.AdamW, "step", phase_timed("opt_s", "opt"))
        patch(torch.Tensor, "cpu", phase_timed("sync_s", "sync", "sync_n"))

    # ------------------------------------------------------------------ tasks
    def begin_task(self, index):
        self.task = index
        if self.first_task is None:
            self.first_task = index
        self.plan_call = 0
        self.task_acc = defaultdict(float)

    def end_task(self, index, result, wall):
        per_level = defaultdict(lambda: {"solves": 0, "iters": 0, "wall_s": 0.0})
        for record in result.get("planning_compute", []) or []:
            entry = per_level[f"L{record.get('level')}"]
            entry["solves"] += 1
            entry["iters"] += sum(record.get("n_iters") or [0])
            entry["wall_s"] += float(record.get("wall_clock") or 0.0)
        self.tasks.append({
            "task": index, "completed": result.get("completed"), "success": result.get("success"),
            "error": result.get("error"), "wall_s": wall,
            "evaluation_wall_seconds": result.get("evaluation_wall_seconds"),
            "peak_allocated_bytes": result.get("peak_allocated_bytes"),
            "acc": dict(self.task_acc), "solve_records": dict(per_level),
        })
        print(f"task {index}: success={result.get('success')} wall={wall:.2f} s "
              f"plan={self.task_acc['plan_s']:.2f} s calls={int(self.task_acc['plan_calls'])} "
              f"env_steps={int(self.task_acc['env_steps'])}", flush=True)

    def dump(self, extra):
        levels = {}
        for key, acc in self.levels.items():
            levels[key] = {field: acc.get(field, 0.0) for field in LEVEL_FIELDS}
        data = {"args": vars(self.args), "levels": levels, "tasks": self.tasks,
                "windows": self.windows, "calls": self.calls, "captured": self.captured, **extra}
        path = self.out / "task_profile.json"
        C.save_json(path, data)
        return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0])
    parser.add_argument("--trace-call", type=int, default=2, help="planning call to profile (0 = off)")
    parser.add_argument("--trace-skip", type=int, default=3, help="GD iterations to skip before the window")
    parser.add_argument("--trace-iters", type=int, default=5)
    parser.add_argument("--no-capture", dest="capture", action="store_false")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = C.out_dir("task_profile")

    import evaluate as E

    t = perf()
    E._worker_init(str(C.ROOT / "hw4.py"), args.device, {"profiling": True}, str(out / "eval"),
                   forward_timing_enabled=False)
    setup_s = perf() - t
    import torch

    model = E._WORKER["model"]
    inst = Instrument(args, out)
    inst.install(model)
    print(f"worker setup {setup_s:.2f} s; tasks {args.tasks}", flush=True)
    for index in args.tasks:
        inst.begin_task(index)
        t = perf()
        result = E._evaluate_task(index)
        inst.end_task(index, result, perf() - t)
    path = inst.dump({"worker_setup_s": setup_s, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name()})
    print(f"task_profile -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
