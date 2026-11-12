"""Short profile of today's model using saved planner inputs, without episodes.

Three representative cases by default. Measures warm production-graph planning
steps separately from eager kernel profiling, including upper projection, cost
and AdamW. The eager inventory avoids missing CUDA graph nodes in CUPTI traces;
its duration percentages are not graph-replay or e1 wall-time percentages.
No model defaults or fixed files are changed. This is a one-worker diagnosis,
not an e1 timing prediction. All output must be outside hw4.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import statistics
import time

import _common as C
from microbench import Bench
from graph_check import clear_graphs, graphs_module


@contextmanager
def annotate(torch, model, cost_model):
    originals = []
    def wrap(obj, name, label):
        original = getattr(obj, name)
        def call(*args, **kwargs):
            with torch.profiler.record_function('prof:' + label):
                return original(*args, **kwargs)
        originals.append((obj, name, original))
        setattr(obj, name, call)
    for level in range(1, model.num_levels + 1):
        wrap(model.get_level(level), 'rollout', 'rollout')
    for name in ('_encode_states_up', '_encode_actions_up'):
        wrap(cost_model, name, 'upper_projection')
    try:
        yield
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def summarize(trace, iterations):
    summary = C.analyze_trace(trace, iterations)
    # _common's historical name is a sum, not a union of overlapping intervals.
    summary['gpu_duration_sum_ms_per_iter'] = summary.pop('gpu_busy_ms_per_iter')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--trace-iters', type=int, default=2)
    args = parser.parse_args()
    paths = sorted(args.inputs.expanduser().glob('*.pt'))
    selected = []
    for level in (3, 2, 1):
        matches = [path for path in paths if f'_L{level}_' in path.name]
        if not matches:
            parser.error(f'no L{level} saved inputs in {args.inputs}')
        selected.append(matches[0])
    out = args.output.expanduser().resolve()
    if out.is_relative_to(C.ROOT.resolve()) or min(args.iters, args.trace_iters) < 1:
        parser.error('output must be outside hw4; iterations must be positive')
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bench = Bench()
    graphs = graphs_module(bench.model)
    if not graphs.ENABLED:
        raise RuntimeError('Run with HW4_ROLLOUT_GRAPH=1 to measure the production graph path')
    report = {'completed': False, 'torch': str(torch.__version__),
              'gpu': torch.cuda.get_device_name(), 'cases': [],
              'scope': 'one worker, saved inputs, warm iterations; excludes startup and capture',
              'source_sha256': {str(p.relative_to(C.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((C.ROOT / 'model').glob('*.py'))}}
    summary_path = out / 'summary.json'
    C.save_json(summary_path, report)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            for path in selected:
                case = bench.load_case(path)
                p = case.actions.detach().clone().requires_grad_(True)
                optimizer = torch.optim.AdamW([p], lr=case.lr)
                cached = dict(case.info)
                def step():
                    info = dict(cached)
                    with torch.profiler.record_function('prof:fwd'):
                        costs = case.cm.get_cost(info, p)
                        for key in ('embed_0', 'goal_embed_0'):
                            if key in info:
                                cached.setdefault(key, info[key])
                    with torch.profiler.record_function('prof:bwd'):
                        costs.sum().backward()
                    with torch.profiler.record_function('prof:opt'):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if case.bounds is not None:
                            with torch.no_grad():
                                p.clamp_(case.bounds[0], case.bounds[1])
                    with torch.profiler.record_function('prof:sync'):
                        costs.min(dim=1).values.detach().cpu().tolist()
                for _ in range(2):
                    step()  # lazy graph capture, optimizer state and cached encodings
                entries = [value for level in range(1, bench.model.num_levels + 1)
                           for value in bench.model.get_level(level).__dict__.get(graphs._CACHE_ATTR, {}).values()]
                if not entries or not all(callable(value) for value in entries):
                    raise RuntimeError(f'{path.name}: production graph capture failed or missing')
                times = []
                for _ in range(args.iters):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    step()
                    torch.cuda.synchronize()
                    times.append(1000 * (time.perf_counter() - started))
                row = {'file': path.name, 'level': case.level, 'horizon': case.horizon,
                       'warm_graph_step_ms': times, 'warm_graph_step_median_ms': statistics.median(times)}
                activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                graphs.ENABLED = False
                try:
                    step()
                    torch.cuda.synchronize()
                    trace = out / f'{path.stem}-eager-operators.json'
                    with annotate(torch, bench.model, case.cm):
                        with torch.profiler.profile(activities=activities) as profile:
                            for _ in range(args.trace_iters):
                                step()
                            torch.cuda.synchronize()
                    profile.export_chrome_trace(str(trace))
                    row['profile'] = summarize(trace, args.trace_iters)
                    row['profile_mode'] = 'eager_operator_inventory'
                    if row['profile']['kernels_per_iter'] == 0:
                        raise RuntimeError('Profiler recorded no CUDA kernels; inspect CUPTI availability')
                finally:
                    graphs.ENABLED = True
                report['cases'].append(row)
                C.save_json(summary_path, report)
                print(f"{path.name}: warm graph step {row['warm_graph_step_median_ms']:.3f} ms; "
                      f"profile={row['profile_mode']}", flush=True)
                categories = row['profile']['categories']
                total = sum(value['ms_per_iter'] for value in categories.values())
                print(C.table(['category', 'GPU duration sum ms/iter', '% of sum', 'count/iter'],
                    [[key, round(value['ms_per_iter'], 4),
                      round(100 * value['ms_per_iter'] / total, 1) if total else None,
                      value['count_per_iter']] for key, value in categories.items()]), flush=True)
                print('phase attribution:', row['profile']['phases'], flush=True)
                clear_graphs(bench.model, graphs)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    else:
        report['completed'] = True
    finally:
        C.save_json(summary_path, report)
        print(f'summary: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
