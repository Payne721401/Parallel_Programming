"""Replay the get_cost inputs captured by task_profile.py, outside the planner.

For each captured (level, horizon, goal space):
  eager_fb    fwd + backward, synchronized              (today's model work)
  eager_full  fwd + bwd + AdamW + clamp + min().cpu()   (one real GD iteration)
  profile     kernels/iter, GPU busy, categories, FLOPs of eager_full
  graph_fb    the same fwd + bwd captured in ONE CUDA graph and replayed (upper bound
              for removing launch overhead without changing a single kernel)
  graph_full  graph replay + the eager AdamW/clamp/min().cpu() that stay in fixed code
  gpu_floor   back-to-back graph replays: pure GPU time per iteration of today's kernels
  exact       max |graph - eager| of cost and action gradient at the same input

--procs N runs eager_full / graph_full / gpu_floor in N processes at once (one key per
level), like evaluate.py's 8 workers sharing the GPU.
Writes $PROF_OUT/microbench.json or microbench_procs<N>.json.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HW4_ASSETS_VERIFIED", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.setup_paths()
perf = time.perf_counter


def median_ms(values):
    return statistics.median(values) * 1000.0 if values else None


class Case:
    pass


class Bench:
    def __init__(self, device="cuda:0", threads=1):
        import torch
        from omegaconf import OmegaConf

        import runner

        self.torch = torch
        torch.set_num_threads(threads)
        torch.cuda.set_device(device)
        runner.add_reference_paths()
        model, _, _ = runner.load_model(submission=C.ROOT / "hw4.py", device=device)
        model.interpolate_pos_encoding = True
        self.model = model
        self.cfg = OmegaConf.load(C.ROOT / "config/fourroom_eval.yaml")
        from hierarchical_solver import LevelCostModel

        levels = [model.get_level(level) for level in range(1, model.num_levels + 1)]
        self.cost_models = {
            level: LevelCostModel(model=levels[level - 1], upper_models=levels[level:])
            for level in range(1, len(levels) + 1)
        }
        self.annotate = False

    def ann(self, name):
        if self.annotate:
            return self.torch.profiler.record_function("prof:" + name)
        return nullcontext()

    def load_case(self, path):
        torch = self.torch
        blob = torch.load(path, map_location="cpu", weights_only=True)
        case = Case()
        case.name = blob["name"]
        case.file = Path(path).name
        case.level = int(blob["level"])
        case.horizon = int(blob["horizon"])
        case.samples = int(blob["num_samples"])
        info = {}
        for key, (kind, tensor) in blob["tensors"].items():
            tensor = tensor.to("cuda")
            if kind == "expand1":
                tensor = tensor.expand(tensor.shape[0], case.samples, *tensor.shape[2:])
            info[key] = tensor
        info.update(blob["meta"])
        case.info = info
        case.actions = blob["actions"].to("cuda")
        case.lr = float(self.cfg.solver.solvers[f"level{case.level}"].optimizer_kwargs.lr)
        case.cm = self.cost_models[case.level]
        queue = case.cm.get_latent_action_queue()
        case.bounds = None
        if torch.is_tensor(queue) and queue.numel() > 0:
            q = queue.detach().to("cuda")
            case.bounds = (torch.quantile(q, 0.02, dim=0).view(1, 1, 1, -1),
                           torch.quantile(q, 0.98, dim=0).view(1, 1, 1, -1))
        return case

    # ----------------------------------------------------------------- eager
    def eager_fb(self, case, n):
        torch = self.torch
        times = []
        for _ in range(n):
            a = case.actions.detach().clone().requires_grad_(True)
            torch.cuda.synchronize()
            t = perf()
            cost = case.cm.get_cost(dict(case.info), a)
            cost.sum().backward()
            torch.cuda.synchronize()
            times.append(perf() - t)
        return times

    def eager_full(self, case, n):
        torch = self.torch
        p = case.actions.detach().clone().requires_grad_(True)
        opt = torch.optim.AdamW([p], lr=case.lr)
        rows = []
        torch.cuda.synchronize()
        for _ in range(n):
            t0 = perf()
            with self.ann("fwd"):
                costs = case.cm.get_cost(dict(case.info), p)
            t1 = perf()
            with self.ann("bwd"):
                costs.sum().backward()
            t2 = perf()
            with self.ann("opt"):
                opt.step()
                opt.zero_grad(set_to_none=True)
                if case.bounds is not None:
                    p.data.clamp_(min=case.bounds[0], max=case.bounds[1])
                best = costs.min(dim=1).values.detach()
            t3 = perf()
            with self.ann("sync"):
                best.cpu().tolist()
            t4 = perf()
            rows.append((t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return rows

    # ----------------------------------------------------------------- graph
    def capture(self, case):
        torch = self.torch
        static_a = case.actions.detach().clone().requires_grad_(True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                static_a.grad = None
                cost = case.cm.get_cost(dict(case.info), static_a)
                cost.sum().backward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        static_a.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_cost = case.cm.get_cost(dict(case.info), static_a)
            static_cost.sum().backward()
        torch.cuda.synchronize()
        case.graph = (graph, static_a, static_cost, static_a.grad)

    def exactness(self, case):
        torch = self.torch
        graph, static_a, static_cost, static_grad = case.graph
        gen = torch.Generator(device="cuda").manual_seed(0)
        x = case.actions + 0.05 * torch.randn(case.actions.shape, generator=gen, device="cuda")
        static_a.data.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
        g_cost, g_grad = static_cost.detach().clone(), static_grad.detach().clone()
        e = x.detach().clone().requires_grad_(True)
        e_cost = case.cm.get_cost(dict(case.info), e)
        e_cost.sum().backward()
        torch.cuda.synchronize()
        return (float((g_cost - e_cost.detach()).abs().max()),
                float((g_grad - e.grad).abs().max()))

    def graph_fb(self, case, n):
        torch = self.torch
        graph = case.graph[0]
        times = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = perf()
            graph.replay()
            torch.cuda.synchronize()
            times.append(perf() - t)
        return times

    def graph_full(self, case, n):
        torch = self.torch
        graph, static_a, static_cost, static_grad = case.graph
        p = case.actions.detach().clone().requires_grad_(True)
        opt = torch.optim.AdamW([p], lr=case.lr)
        times = []
        torch.cuda.synchronize()
        for _ in range(n):
            t = perf()
            static_a.data.copy_(p.detach())
            graph.replay()
            p.grad = static_grad
            opt.step()
            opt.zero_grad(set_to_none=True)
            if case.bounds is not None:
                p.data.clamp_(min=case.bounds[0], max=case.bounds[1])
            static_cost.min(dim=1).values.detach().cpu().tolist()
            times.append(perf() - t)
        return times

    def gpu_floor(self, case, n):
        torch = self.torch
        graph = case.graph[0]
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        begin.record()
        for _ in range(n):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return begin.elapsed_time(end) / n

    def profile_eager(self, case, out, n=5):
        torch = self.torch
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        self.annotate = True
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, with_flops=True) as prof:
                self.eager_full(case, n)
                torch.cuda.synchronize()
        finally:
            self.annotate = False
        trace = out / "microbench_traces" / f"{case.file[:-3]}.json"
        trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace))
        stats = C.analyze_trace(trace, n)
        try:
            stats["gflop_per_iter"] = sum(
                float(getattr(e, "flops", 0) or 0) for e in prof.key_averages()
            ) / 1e9 / n
        except Exception as error:
            stats["flops_error"] = repr(error)
        return stats


def input_files(args):
    folder = Path(args.inputs) if args.inputs else C.out_dir("task_profile", "inputs")
    files = sorted(folder.glob("*.pt"))
    if not files:
        raise SystemExit(f"No captured inputs in {folder}; run prof/task_profile.py first.")
    return files


def first_per_level(files):
    chosen = {}
    for path in files:
        level = path.name.split("_")[1]  # NN_L3_H6_...
        chosen.setdefault(level, path)
    return [chosen[k] for k in sorted(chosen, reverse=True)]


def run_single(args):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch

    out = C.out_dir()
    bench = Bench()
    results = []
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        for path in input_files(args):
            case = bench.load_case(path)
            row = {"name": case.name, "file": case.file, "level": case.level,
                   "horizon": case.horizon, "samples": case.samples}
            print(f"[{case.file}] L{case.level} H{case.horizon} S{case.samples}", flush=True)
            try:
                bench.eager_fb(case, 3)
                row["eager_fb_ms"] = median_ms(bench.eager_fb(case, args.iters))
                rows = bench.eager_full(case, args.iters)
                row["eager_full_ms"] = median_ms([r[0] for r in rows])
                row["host_fwd_ms"] = median_ms([r[1] for r in rows])
                row["host_bwd_ms"] = median_ms([r[2] for r in rows])
                row["host_rest_ms"] = median_ms([r[3] for r in rows])
                row["sync_wait_ms"] = median_ms([r[4] for r in rows])
            except Exception:
                row["eager_error"] = traceback.format_exc(limit=3)
            try:
                row["profile"] = bench.profile_eager(case, out)
            except Exception:
                row["profile_error"] = traceback.format_exc(limit=3)
            try:
                bench.capture(case)
                row["graph_max_abs_cost"], row["graph_max_abs_grad"] = bench.exactness(case)
                bench.graph_fb(case, 5)
                row["graph_fb_ms"] = median_ms(bench.graph_fb(case, args.graph_iters))
                bench.graph_full(case, 5)
                row["graph_full_ms"] = median_ms(bench.graph_full(case, args.graph_iters))
                row["gpu_floor_ms"] = bench.gpu_floor(case, args.graph_iters)
            except Exception:
                row["graph_error"] = traceback.format_exc(limit=4)
            case.graph = None
            print("   " + ", ".join(f"{k}={C.fmt(v, 3)}" for k, v in row.items()
                                    if k.endswith("_ms") or k.startswith("graph_max")), flush=True)
            results.append(row)
            torch.cuda.empty_cache()
    data = {"mode": "single", "results": results, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name()}
    C.save_json(out / "microbench.json", data)
    print(f"microbench -> {out / 'microbench.json'}")


def child(rank, nprocs, barrier, queue, files, args):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import torch

        bench = Bench()
        results = {}
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            cases = [bench.load_case(path) for path in files]
            for case in cases:
                bench.eager_full(case, 3)
                case.graph = None
                case.graph_error = None
                try:
                    bench.capture(case)
                    bench.graph_full(case, 3)
                except Exception:
                    case.graph = None
                    case.graph_error = traceback.format_exc(limit=3)
            torch.cuda.synchronize()
            # Every process passes the same barriers, even when its graph capture failed.
            for case in cases:
                entry = {"graph_error": case.graph_error}
                barrier.wait(timeout=600)
                t = perf()
                rows = bench.eager_full(case, args.iters)
                entry["eager_phase_s"] = perf() - t
                entry["eager_full_ms"] = median_ms([r[0] for r in rows])
                entry["host_fwd_ms"] = median_ms([r[1] for r in rows])
                entry["host_bwd_ms"] = median_ms([r[2] for r in rows])
                entry["sync_wait_ms"] = median_ms([r[4] for r in rows])
                barrier.wait(timeout=600)
                if case.graph is not None:
                    t = perf()
                    entry["graph_full_ms"] = median_ms(bench.graph_full(case, args.graph_iters))
                    entry["graph_phase_s"] = perf() - t
                barrier.wait(timeout=600)
                if case.graph is not None:
                    t = perf()
                    entry["gpu_floor_ms"] = bench.gpu_floor(case, args.graph_iters)
                    entry["floor_phase_s"] = perf() - t
                results[case.name] = entry
        queue.put(("ok", rank, results))
    except Exception:
        try:
            barrier.abort()
        except Exception:
            pass
        queue.put(("error", rank, traceback.format_exc()))


def run_procs(args):
    import multiprocessing as mp

    files = first_per_level(input_files(args))
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.procs)
    queue = ctx.Queue()
    procs = [ctx.Process(target=child, args=(rank, args.procs, barrier, queue, files, args))
             for rank in range(args.procs)]
    started = perf()
    for proc in procs:
        proc.start()
    messages = []
    for _ in procs:
        try:
            messages.append(queue.get(timeout=420))
        except Exception:
            messages.append(("error", -1, "a child did not report within 420 s"))
            break
    for proc in procs:
        proc.join(timeout=60)
    errors = [m for m in messages if m[0] != "ok"]
    for _, rank, text in errors:
        print(f"rank {rank} failed:\n{text}", flush=True)
    per_case = {}
    for status, rank, results in messages:
        if status != "ok":
            continue
        for name, entry in results.items():
            per_case.setdefault(name, []).append(entry)
    summary = {}
    for name, entries in per_case.items():
        agg = {"procs": len(entries)}
        for field in ("eager_full_ms", "host_fwd_ms", "host_bwd_ms", "sync_wait_ms",
                      "graph_full_ms", "gpu_floor_ms"):
            values = [e[field] for e in entries if e.get(field) is not None]
            agg[field + "_mean"] = statistics.mean(values) if values else None
            agg[field + "_max"] = max(values) if values else None
        eager_walls = [e["eager_phase_s"] for e in entries if e.get("eager_phase_s")]
        graph_walls = [e["graph_phase_s"] for e in entries if e.get("graph_phase_s")]
        agg["eager_aggregate_iters_per_s"] = (len(eager_walls) * args.iters / max(eager_walls)
                                              if eager_walls else None)
        agg["graph_aggregate_iters_per_s"] = (len(graph_walls) * args.graph_iters / max(graph_walls)
                                              if graph_walls else None)
        errors_seen = [e["graph_error"] for e in entries if e.get("graph_error")]
        if errors_seen:
            agg["graph_error"] = errors_seen[0]
        summary[name] = agg
        print(f"{name}: " + ", ".join(f"{k}={C.fmt(v, 3)}" for k, v in agg.items()), flush=True)
    out = C.out_dir()
    data = {"mode": f"procs{args.procs}", "procs": args.procs, "files": [p.name for p in files],
            "summary": summary, "errors": [m[2] for m in errors],
            "wall_s": perf() - started, "iters": args.iters, "graph_iters": args.graph_iters}
    path = out / f"microbench_procs{args.procs}.json"
    C.save_json(path, data)
    print(f"microbench -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=None, help="folder of captured *.pt (default from PROF_OUT)")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--graph-iters", type=int, default=50)
    parser.add_argument("--procs", type=int, default=1)
    args = parser.parse_args()
    if args.procs > 1:
        run_procs(args)
    else:
        run_single(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
