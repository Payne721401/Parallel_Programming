"""Bounded production-graph worker scaling and Nsight targets on one saved case.

All workers stay resident. Only the selected active workers run GPU work, so
1/2/4/8 compares steady-state contention at a fixed resident memory footprint.
No episodes, no worker-count changes to evaluate.py, no MPS configuration changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import _common as C


class Work:
    def __init__(self, path, graph=True):
        import torch
        from microbench import Bench
        from graph_check import graphs_module
        self.torch = torch
        self.bench = Bench()
        self.graphs = graphs_module(self.bench.model)
        self.graphs.ENABLED = graph
        self.case = self.bench.load_case(path)
        self.p = self.case.actions.detach().clone().requires_grad_(True)
        self.info = dict(self.case.info)
        self.reset()
        self.step()
        self.step()
        torch.cuda.synchronize()
        if graph:
            entries = [v for level in range(1, self.bench.model.num_levels + 1)
                       for v in self.bench.model.get_level(level).__dict__.get(self.graphs._CACHE_ATTR, {}).values()]
            if not entries or not all(callable(v) for v in entries):
                raise RuntimeError('Missing or failed production rollout graph')

    def reset(self):
        with self.torch.no_grad():
            self.p.copy_(self.case.actions)
        self.p.grad = None
        self.optimizer = self.torch.optim.AdamW([self.p], lr=self.case.lr)

    def forward(self):
        info = dict(self.info)
        costs = self.case.cm.get_cost(info, self.p)
        for key in ('embed_0', 'goal_embed_0'):
            if key in info:
                self.info.setdefault(key, info[key])
        return costs

    def step(self):
        torch = self.torch
        with torch.cuda.nvtx.range('diag_forward'):
            costs = self.forward()
        with torch.cuda.nvtx.range('diag_backward'):
            costs.sum().backward()
        with torch.cuda.nvtx.range('diag_optimizer'):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.case.bounds is not None:
                with torch.no_grad():
                    self.p.clamp_(self.case.bounds[0], self.case.bounds[1])
        with torch.cuda.nvtx.range('diag_sync'):
            costs.min(dim=1).values.detach().cpu().tolist()


def worker(rank, connection, start_event, path):
    try:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            work = Work(path)
            connection.send({'kind': 'initialized', 'rank': rank, 'pid': os.getpid(),
                'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(), 'torch': str(torch.__version__)})
            while True:
                command = connection.recv()
                if command[0] == 'stop':
                    break
                if command[0] in ('profile_start', 'profile_stop'):
                    getattr(torch.cuda.profiler, command[0].split('_')[1])()
                    connection.send({'kind': 'profile_ack', 'rank': rank})
                    continue
                _, nactive, iterations = command
                work.reset()
                work.step()  # fresh optimizer state, outside the measured range
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                connection.send({'kind': 'ready', 'rank': rank})
                if not start_event.wait(120):
                    raise TimeoutError('parent did not release the phase')
                durations = []
                started = time.perf_counter_ns()
                with torch.cuda.nvtx.range(f'diag_active{nactive}_rank{rank}'):
                    for _ in range(iterations):
                        t = time.perf_counter_ns()
                        work.step()
                        durations.append((time.perf_counter_ns() - t) / 1e6)
                    torch.cuda.synchronize()
                ended = time.perf_counter_ns()
                connection.send({'kind': 'result', 'rank': rank, 'pid': os.getpid(),
                    'start_ns': started, 'end_ns': ended,
                    'step_ms_median': statistics.median(durations),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    except BaseException:
        try:
            connection.send({'kind': 'error', 'rank': rank, 'error': traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def receive(connection, expected, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not connection.poll(remaining):
        raise TimeoutError(f'timed out waiting for {expected}')
    result = connection.recv()
    if result['kind'] != expected:
        raise RuntimeError(str(result))
    return result


def summarize_phase(rows, iterations):
    elapsed = (max(row['end_ns'] for row in rows) - min(row['start_ns'] for row in rows)) / 1e9
    skew = (max(row['start_ns'] for row in rows) - min(row['start_ns'] for row in rows)) / 1e6
    return {'active_workers': len(rows), 'iterations_per_worker': iterations,
            'phase_wall_s': elapsed, 'start_skew_ms': skew,
            'aggregate_iterations_per_s': len(rows) * iterations / elapsed,
            'worker_step_median_ms_mean': statistics.mean(row['step_ms_median'] for row in rows),
            'workers': rows}


def device_snapshot():
    result = {}
    commands = {
        'gpu': ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm', '--format=csv'],
        'processes': ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'],
        'mps_servers': ['nvidia-cuda-mps-control'],
    }
    for name, command in commands.items():
        try:
            completed = subprocess.run(command, input='get_server_list\n' if name == 'mps_servers' else None,
                                       capture_output=True, text=True, timeout=3)
            result[name] = {'returncode': completed.returncode, 'stdout': completed.stdout,
                            'stderr': completed.stderr}
            if name == 'mps_servers' and completed.returncode == 0:
                for server in completed.stdout.split():
                    if server.isdigit():
                        clients = subprocess.run(command, input=f'get_client_list {server}\n',
                                                 capture_output=True, text=True, timeout=3)
                        result[f'mps_clients_{server}'] = {'returncode': clients.returncode,
                            'stdout': clients.stdout, 'stderr': clients.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {'unavailable': str(error)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--mode', choices=('scaling', 'trace', 'kernel'), default='scaling')
    args = parser.parse_args()
    if not args.input.is_file() or args.iters < 1:
        parser.error('input file must exist; iters must be positive')
    out = args.output.expanduser().resolve()
    if out.is_relative_to(C.ROOT.resolve()):
        parser.error('output must be outside hw4')
    out.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'mode': args.mode, 'input': str(args.input),
              'resident_workers': args.workers if args.mode != 'kernel' else 1,
              'phases': [], 'environment': {k: os.environ.get(k) for k in (
                  'CUDA_VISIBLE_DEVICES', 'CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')},
              'source_sha256': {str(p.relative_to(C.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((C.ROOT / 'model').glob('*.py'))}}
    processes, connections = [], []
    summary = out / 'summary.json'
    C.save_json(summary, report)
    try:
        if args.mode == 'kernel':
            import torch
            from torch.nn.attention import SDPBackend, sdpa_kernel
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with sdpa_kernel(SDPBackend.MATH):
                work = Work(args.input, graph=False)
                torch.cuda.profiler.start()
                try:
                    work.step()
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.profiler.stop()
            report['note'] = 'one eager steady-state step for single-kernel counters, not throughput'
        else:
            ctx = mp.get_context('spawn')
            event = ctx.Event()
            for rank in range(args.workers):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=worker, args=(rank, child, event, args.input))
                process.start()
                child.close()
                processes.append(process)
                connections.append(parent)
            deadline = time.monotonic() + 120
            report['initialized_workers'] = [receive(conn, 'initialized', deadline) for conn in connections]
            report['all_workers_resident_snapshot'] = device_snapshot()
            C.save_json(summary, report)
            order = [n for n in (1, 2, 4, 8) if n <= args.workers]
            order += list(reversed(order))
            if args.mode == 'trace':
                order = [args.workers]
                connections[0].send(('profile_start',))
                receive(connections[0], 'profile_ack', time.monotonic() + 30)
            for active in order:
                event.clear()
                for conn in connections[:active]:
                    conn.send(('run', active, args.iters))
                deadline = time.monotonic() + 120
                for conn in connections[:active]:
                    receive(conn, 'ready', deadline)
                event.set()
                rows = [receive(conn, 'result', deadline) for conn in connections[:active]]
                phase = summarize_phase(rows, args.iters)
                report['phases'].append(phase)
                C.save_json(summary, report)
                print({key: value for key, value in phase.items() if key != 'workers'}, flush=True)
            if args.mode == 'trace':
                connections[0].send(('profile_stop',))
                receive(connections[0], 'profile_ack', time.monotonic() + 30)
            else:
                grouped = {n: statistics.mean(p['aggregate_iterations_per_s'] for p in report['phases']
                                              if p['active_workers'] == n) for n in sorted(set(order))}
                report['scaling'] = [{'active_workers': n, 'aggregate_iterations_per_s': rate,
                                      'throughput_vs_one': rate / grouped[1]} for n, rate in grouped.items()]
                print(json.dumps(report['scaling'], indent=2), flush=True)
        report['completed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        for conn in connections:
            try:
                conn.send(('stop',))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in processes:
            process.join(timeout=2)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for conn in connections:
            conn.close()
        C.save_json(summary, report)
        print(f'summary: {summary}', flush=True)


if __name__ == '__main__':
    main()
