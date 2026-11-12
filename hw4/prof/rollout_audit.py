"""Audit slice/unbind backward on the current rollout without editing model code.

CPU uses synthetic frozen weights. --assets uses the released model on CUDA;
--inputs optionally uses captured planner inputs instead of random inputs.
Reports operator counts separately from unprofiled CUDA graph replay timings.
This is a rollout probe, not an e1 performance or task-success test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import functools
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def candidate(module):
    source = inspect.getsource(module.rollout)
    old = 'conditions.append((shift_msa, 1 + scale_msa, gate_msa, shift_mlp, 1 + scale_mlp, gate_mlp))'
    new = 'conditions.append(tuple(value.unbind(dim=1) for value in (shift_msa, 1 + scale_msa, gate_msa, shift_mlp, 1 + scale_mlp, gate_mlp)))'
    if source.count(old) != 1:
        raise RuntimeError('Rollout source changed; review the audit transformation')
    changed = source.replace(old, new)
    for name in ('shift_msa', 'scale_msa', 'gate_msa', 'shift_mlp', 'scale_mlp', 'gate_mlp'):
        old = f'{name}[:, t]'
        if changed.count(old) != 1:
            raise RuntimeError(f'Expected exactly one {old}')
        changed = changed.replace(old, f'{name}[t]')
    namespace = dict(vars(module))
    exec(compile(changed, '<audit-unbind-candidate>', 'exec'), namespace)
    return namespace['rollout'], source, changed


def node_counts(output):
    pending, seen, counts = [output.grad_fn], set(), Counter()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        counts[type(node).__name__] += 1
        pending.extend(parent for parent, _ in node.next_functions)
    return dict(sorted(counts.items()))


def synthetic(torch, device, level):
    from model.jepa import JEPA, ProjectedPredictor
    from model.modules import Embedder, MLP
    from model.predictors import ARPredictor

    predictor = ARPredictor(num_frames=3, depth=4, heads=8, dim_head=64,
                            mlp_dim=768, input_dim=192, hidden_dim=192)
    # The constructor zeroes adaLN. Nonzero frozen test weights exercise every branch.
    for block in predictor.transformer.layers:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
    action_dim = 10 if level == 1 else 4
    model = JEPA(torch.nn.Identity(), ProjectedPredictor(predictor,
                 MLP(192, 384, norm_fn=torch.nn.BatchNorm1d)),
                 Embedder(action_dim, action_dim, 192, patch_embed=level == 1), level=level)
    return model.to(device).eval().requires_grad_(False), action_dim


def maxdiff(torch, a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError('shape/dtype mismatch')
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise AssertionError('non-finite value')
    return float((a - b).abs().max())


def probe(torch, fn, emb, actions, cotangent, trace):
    a = actions.detach().clone().requires_grad_(True)
    output = fn(emb, a)
    nodes = node_counts(output)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if a.is_cuda:
        torch.cuda.synchronize()
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        gradient, = torch.autograd.grad(output, a, cotangent)
        if a.is_cuda:
            torch.cuda.synchronize()
    profile.export_chrome_trace(str(trace))
    counts = {event.key: event.count for event in profile.key_averages()
              if event.key.startswith('aten::')}
    device_events = [event for event in profile.events()
                     if str(event.device_type).endswith('CUDA')]
    return output.detach(), gradient.detach(), {
        'autograd_nodes': nodes, 'backward_aten_calls': counts,
        'backward_cuda_events': len(device_events),
        'backward_cuda_event_sum_us': sum(event.device_time_total for event in device_events),
    }


def graph_benchmark(torch, functions, emb, actions, cotangent, repeats):
    callables, captures, capture_diffs = {}, {}, {}
    for name, fn in functions.items():
        a = actions.detach().clone().requires_grad_(True)
        sample = (emb.clone(), a)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        graphed = torch.cuda.make_graphed_callables(fn, sample, num_warmup_iters=1)
        torch.cuda.synchronize()
        captures[name] = {'seconds': time.perf_counter() - started,
                          'allocated_delta_bytes': torch.cuda.memory_allocated() - before}
        expected = fn(emb, a)
        eg, = torch.autograd.grad(expected, a, cotangent)
        actual = graphed(emb, a)
        ag, = torch.autograd.grad(actual, a, cotangent)
        capture_diffs[name] = {'output_max_abs': maxdiff(torch, expected, actual),
                               'gradient_max_abs': maxdiff(torch, eg, ag)}
        callables[name] = graphed, a
    rows = []
    order = list(functions) + list(reversed(functions))
    for name in order:
        fn, a = callables[name]
        for _ in range(3):
            torch.autograd.grad(fn(emb, a), a, cotangent)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(repeats):
            torch.autograd.grad(fn(emb, a), a, cotangent)
        end.record()
        end.synchronize()
        rows.append({'mode': name, 'gpu_ms_per_call': start.elapsed_time(end) / repeats,
                     'wall_ms_per_call': 1000 * (time.perf_counter() - wall) / repeats})
    means = {name: statistics.mean(row['gpu_ms_per_call'] for row in rows if row['mode'] == name)
             for name in functions}
    return {'captures': captures, 'capture_differences': capture_diffs, 'runs': rows,
            'gpu_ms_mean': means, 'slice_to_unbind_speedup': means['slice'] / means['unbind']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--assets', action='store_true')
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--reference-loop', action='store_true',
                        help='also profile/time the original unbatched loop with V-only attention')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if out.is_relative_to(ROOT) or args.repeats < 1:
        parser.error('output must be outside hw4 and repeats must be positive')
    if args.inputs and not args.assets:
        parser.error('--inputs requires --assets')
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from model import fast_rollout
    from model import modules

    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    modules.SINGLE_TOKEN_ATTENTION = 'v'
    alternative, source, changed = candidate(fast_rollout)
    (out / 'slice_source.py').write_text(source, encoding='utf-8')
    (out / 'unbind_candidate.py').write_text(changed, encoding='utf-8')
    report = {'torch': str(torch.__version__), 'device': args.device, 'assets': args.assets,
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'results': [],
              'attention': 'v', 'reference_loop': args.reference_loop,
              'model_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((ROOT / 'model').glob('*.py'))},
              'scope': 'single-process rollout; no e1 timing or success claim'}
    if args.device == 'cuda':
        report['gpu'] = torch.cuda.get_device_name()
    model = architecture = None
    if args.assets:
        import runner
        model, architecture, _ = runner.load_model(device=args.device)
    cases = []
    if args.inputs:
        for path in sorted(args.inputs.glob('*.pt')):
            blob = torch.load(path, map_location=args.device, weights_only=True)
            if 'embed_0' not in blob['tensors']:
                raise RuntimeError(f'{path}: missing captured embed_0')
            kind, emb = blob['tensors']['embed_0']
            # Both expand1 storage and materialized inputs have the sample axis.
            emb = emb[:, 0]
            actions = blob['actions']
            if emb.shape[1] != 1:
                raise RuntimeError(f'{path}: only one-token history is supported')
            cases.append((path.stem, model.get_level(int(blob['level'])), emb, actions))
        if not cases:
            raise RuntimeError('No captured input files found')
    else:
        for level, samples, steps in ((3, 200, 6), (2, 200, 2), (1, 50, 2)):
            if args.assets:
                jepa = model.get_level(level)
                action_dim = architecture['levels'][level - 1]['action_embed']['input_dim']
            else:
                jepa, action_dim = synthetic(torch, args.device, level)
            cases.append((f'L{level}-H{steps}', jepa,
                          torch.randn(1, 1, 192, device=args.device),
                          torch.randn(1, samples, steps, action_dim, device=args.device)))
    try:
        with sdpa_kernel(SDPBackend.MATH):
            for name, jepa, emb, actions in cases:
                if not fast_rollout.supported(jepa, 1, 1):
                    raise RuntimeError(f'{name}: current fast rollout is disabled or unsupported')
                functions = {'slice': functools.partial(fast_rollout.rollout, jepa),
                             'unbind': functools.partial(alternative, jepa)}
                if args.reference_loop:
                    def original_loop(e, a, level=jepa):
                        enabled = fast_rollout.ENABLED
                        fast_rollout.ENABLED = False
                        try:
                            return level._rollout_core(e, a, 1, 1)
                        finally:
                            fast_rollout.ENABLED = enabled
                    functions = {'loop': original_loop, **functions}
                shape = (actions.shape[0], actions.shape[1], actions.shape[2] + 1, emb.shape[-1])
                cotangent = torch.randn(shape, device=args.device)
                values, row = {}, {'name': name, 'action_shape': list(actions.shape)}
                for mode, fn in functions.items():
                    output, gradient, details = probe(torch, fn, emb, actions, cotangent,
                                                       out / f'{name}-{mode}-backward.json')
                    values[mode] = output, gradient
                    row[mode] = details
                row['output_max_abs'] = maxdiff(torch, values['slice'][0], values['unbind'][0])
                row['gradient_max_abs'] = maxdiff(torch, values['slice'][1], values['unbind'][1])
                if args.reference_loop:
                    row['loop_to_slice_output_max_abs'] = maxdiff(torch, values['loop'][0], values['slice'][0])
                    row['loop_to_slice_gradient_max_abs'] = maxdiff(torch, values['loop'][1], values['slice'][1])
                if args.device == 'cuda':
                    row['graph'] = graph_benchmark(torch, functions, emb, actions, cotangent, args.repeats)
                report['results'].append(row)
                print(json.dumps({key: value for key, value in row.items() if key not in functions}), flush=True)
                for mode in functions:
                    counts = row[mode]['backward_aten_calls']
                    print(mode, {key: counts.get(key, 0) for key in (
                        'aten::select_backward', 'aten::slice_backward', 'aten::zero_',
                        'aten::add_', 'aten::stack', 'aten::mm')}, flush=True)
        report['completed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'summary: {out / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
