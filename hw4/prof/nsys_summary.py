"""Summarize device kernel intervals; no CUDA or third-party dependencies.

Uses Nsight's documented SQLite schema. Overlap is across OS processes on the
same device, not CPU ranges or streams. Profiling changes execution timing.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3


def interval_stats(rows):
    events = defaultdict(list)
    for start, end, pid in rows:
        if end > start:
            events[start].append((pid, 1))
            events[end].append((pid, -1))
    active = Counter()
    busy = overlap = 0
    previous = None
    for at, changes in sorted(events.items()):
        if previous is not None:
            busy += (at - previous) if active else 0
            overlap += (at - previous) if len(active) > 1 else 0
        for pid, delta in changes:
            active[pid] += delta
            if not active[pid]:
                del active[pid]
        previous = at
    return busy, overlap


def analyze(path, expected_workers):
    with closing(sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)) as db:
        return analyze_db(db, expected_workers)


def analyze_db(db, expected_workers):
    db.row_factory = sqlite3.Row
    columns = {r[1] for r in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    required = {'start', 'end', 'globalPid', 'deviceId', 'demangledName'}
    if not required <= columns:
        raise ValueError(f'Missing kernel columns: {required - columns}; node tracing may be unavailable')
    names = dict(db.execute('SELECT id,value FROM StringIds'))
    kernels = list(db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'))
    if not kernels:
        raise ValueError('No GPU kernels recorded; this is not a successful timeline')
    # Use the intersection of worker NVTX ranges when the schema provides them.
    # The fallback is explicitly labelled and includes pre-phase warmup work.
    ranges = []
    nvcols = {r[1] for r in db.execute('PRAGMA table_info(NVTX_EVENTS)')}
    if {'start', 'end', 'globalTid'} <= nvcols:
        for r in db.execute('SELECT * FROM NVTX_EVENTS WHERE end > start'):
            label = (r['text'] if 'text' in nvcols else None) or names.get(r['textId'] if 'textId' in nvcols else None, '')
            if label.startswith('diag_active'):
                ranges.append((r['start'], r['end'], (r['globalTid'] // 0x1000000) % 0x1000000))
    window = None
    if len({r[2] for r in ranges}) == expected_workers:
        lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
        if hi > lo:
            window = (lo, hi)
    devices = defaultdict(list)
    grouped = defaultdict(lambda: [0, 0])
    contexts = set()
    for r in kernels:
        start, end = r['start'], r['end']
        if window:
            start, end = max(start, window[0]), min(end, window[1])
        if end <= start:
            continue
        pid = (r['globalPid'] // 0x1000000) % 0x1000000
        devices[r['deviceId']].append((start, end, pid))
        contexts.add((pid, r['contextId'] if 'contextId' in columns else None))
        name = names.get(r['demangledName'], str(r['demangledName']))
        grouped[name][0] += end - start
        grouped[name][1] += 1
    ranked = [{'name': n, 'duration_sum_ms': v[0] / 1e6, 'count': v[1]}
              for n, v in sorted(grouped.items(), key=lambda item: item[1][0], reverse=True)]
    per_device = []
    for device, rows in devices.items():
        busy, overlap = interval_stats(rows)
        span = window[1] - window[0] if window else max(r[1] for r in rows) - min(r[0] for r in rows)
        per_device.append({'device': device, 'pids': sorted({r[2] for r in rows}),
                           'window_ms': span / 1e6, 'kernel_union_ms': busy / 1e6,
                           'kernel_union_pct_of_window': 100 * busy / span,
                           'cross_process_overlap_ms': overlap / 1e6,
                           'cross_process_overlap_pct_of_kernel_union': 100 * overlap / busy})
    pids = {pid for rows in devices.values() for _, _, pid in rows}
    return {'all_expected_workers_observed': len(pids) == expected_workers,
            'window': 'common active worker NVTX interval' if window else 'whole captured kernel span (includes warmup)',
            'contexts': sorted(contexts), 'devices': per_device,
            'top_kernels': ranked[:20],
            'gemm_targets': [r for r in ranked if re.search('gemm|gemv|cutlass|xmma', r['name'], re.I)][:2],
            'caveat': 'Device kernel intervals under profiler; union is not SM utilization. Only draw a multiprocess conclusion if all expected workers were recorded.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    report = analyze(args.sqlite, args.workers)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for key in ('all_expected_workers_observed', 'window', 'devices', 'gemm_targets', 'caveat'):
        print(f'{key}: {json.dumps(report[key])}')
    for i, target in enumerate(report['gemm_targets'], 1):
        # A single literal name, never shell-evaluated. NCU demangled-name filter.
        args.output.with_name(f'gemm-filter-{i}.txt').write_text('regex:^' + re.escape(target['name']) + '$', encoding='utf-8')
    if not report['all_expected_workers_observed']:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
