#!/usr/bin/env python3
"""Compare tile/bbox ABBA; work may change across methods but not repetitions."""
import csv
import math
from pathlib import Path
import statistics
import sys

from run_cases import SPARSE_COLUMNS


def verify_dispatch(text, method):
    lines = [line for line in text.splitlines() if line.startswith('[PP_SUPPORT] ')]
    if len(lines) != 1:
        raise ValueError('missing/duplicate support diagnostic; rebuild the binary')
    fields = dict(token.split('=', 1) for token in lines[0].split()[1:])
    if fields != dict(mode=method, selected=str(int(method == 'bbox')), initial='tile'):
        raise ValueError('wrong support dispatch: ' + str(fields))


def summarize(root, profiled):
    samples = {}
    for round_id, method in ((1,'tile'),(2,'bbox'),(3,'bbox'),(4,'tile')):
        folder = root/f'{round_id}-{method}'
        with (folder/'results.tsv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        if len(rows) != 3 or {Path(r['case']).stem for r in rows} != {'08','09','10'}:
            raise ValueError('expected exactly 08/09/10 in each round')
        for index, row in enumerate(rows):
            case = Path(row['case']).stem
            if row['verdict'] not in ('AC','TLE'):
                raise ValueError('unverified output: ' + case)
            value = float(row['compute_s'])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid Compute time')
            if profiled:
                verify_dispatch((folder/f'{index:02d}-{case}.log').read_text(encoding='utf-8'), method)
                expected_kernel = 'sparse_sliding' if case == '09' else 'sparse_r1'
                if (row['profile_status'] != 'ok' or row['profile_kernel'] != expected_kernel
                        or row['sparse_status'] != 'ok' or row['sparse_selected'] != '1'):
                    raise ValueError('unexpected profile/scheduler: ' + case)
                for key in ('kernels_ms','sparse_mean_work'):
                    n = float(row[key])
                    if not math.isfinite(n) or n < 0:
                        raise ValueError('invalid '+key)
            samples.setdefault((case,method),[]).append(row)
    output = []
    for case in ('08','09','10'):
        a,b = samples[case,'tile'],samples[case,'bbox']
        if profiled:
            for group in (a,b):
                if len({tuple(r[k] for k in SPARSE_COLUMNS) for r in group}) != 1:
                    raise ValueError('non-deterministic work within one method: ' + case)
            for key in ('sparse_tiles_total','sparse_initial_work'):
                if len({r[key] for r in a+b}) != 1:
                    raise ValueError('initial geometry/schedule changed: ' + case)
        ta,tb = [[float(r['compute_s']) for r in group] for group in (a,b)]
        ma,mb = statistics.mean(ta),statistics.mean(tb)
        row = dict(case=case,tile_mean_s=ma,bbox_mean_s=mb,tile_over_bbox=ma/mb,
                   tile_min_s=min(ta),tile_max_s=max(ta),bbox_min_s=min(tb),bbox_max_s=max(tb))
        print(f'{case}: tile {ma:.6f}s -> bbox {mb:.6f}s; {ma/mb:.3f}x; '
              f'ranges {min(ta):.6f}..{max(ta):.6f} / {min(tb):.6f}..{max(tb):.6f}')
        if profiled:
            for method,group in (('tile',a),('bbox',b)):
                row[method+'_kernels_ms'] = statistics.mean(float(r['kernels_ms']) for r in group)
                row[method+'_mean_work'] = float(group[0]['sparse_mean_work'])
                row[method+'_dense_steps'] = int(group[0]['sparse_dense_steps'])
            print(f"  work/step {row['tile_mean_work']:.2f} -> {row['bbox_mean_work']:.2f}; "
                  f"kernel interval {row['tile_kernels_ms']:.3f} -> {row['bbox_kernels_ms']:.3f} ms; "
                  f"dense steps {row['tile_dense_steps']} -> {row['bbox_dense_steps']}")
        output.append(row)
    with (root/'summary.tsv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]),delimiter='\t')
        writer.writeheader()
        writer.writerows(output)
    print('Two samples/method; work reduction is not speedup. No default changed.')


if __name__ == '__main__':
    try:
        if len(sys.argv)!=3 or sys.argv[2] not in ('0','1'):
            raise ValueError('usage: summarize_support.py LOGDIR PROFILE_0_OR_1')
        summarize(Path(sys.argv[1]),sys.argv[2]=='1')
    except (OSError,ValueError,KeyError) as error:
        print('support summary: '+str(error),file=sys.stderr)
        sys.exit(1)
