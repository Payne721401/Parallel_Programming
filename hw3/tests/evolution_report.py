#!/usr/bin/env python3
"""Validate exact evolution counters and produce diagnostic TSVs, never GPU work."""
import argparse
import csv
import math
from pathlib import Path
import sys

COUNTS = ('alive nonzero decay changed nonzero_tiles changed_tiles influence_tiles work_tiles '
          'empty_tiles tiles_1_16 tiles_17_128 tiles_129_512 tiles_513_up').split()
BINS = COUNTS[-5:]


def fields(line):
    return dict(token.split('=', 1) for token in line.split()[1:])


def schedule(steps, every):
    return [t for t in range(steps+1) if t in (0, steps) or
            (t <= 64 and t > 0 and t & (t-1) == 0) or t % every == 0]


def parse(text):
    rows, metas = [], []
    for line in text.splitlines():
        if line.startswith('[PP_EVOLVE] '):
            row = fields(line)
            for key in ['generation', *COUNTS, 'scheduled_next', 'dense', 'fixed']:
                row[key] = int(row[key])
                if row[key] < 0:
                    raise ValueError('negative evolution count')
            for key in ('scan_ms', 'update_ms'):
                row[key] = float(row[key])
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError('invalid evolution duration')
            rows.append(row)
        elif line.startswith('[PP_EVOLVE_META] '):
            metas.append(fields(line))
    if len(metas) != 1:
        raise ValueError('expected one evolution metadata record')
    meta = metas[0]
    for key in ('steps', 'cells', 'tiles', 'samples', 'every', 'sparse', 'first_observed_fixed'):
        meta[key] = int(meta[key])
    for key in ('setup_host_ms', 'scans_ms', 'collect_host_ms', 'free_host_ms'):
        meta[key] = float(meta[key])
        if not math.isfinite(meta[key]) or meta[key] < 0:
            raise ValueError('invalid evolution metadata duration')
    if (meta['steps'] < 0 or meta['cells'] <= 0 or meta['tiles'] <= 0 or
            not 1 <= meta['every'] <= 5000 or meta['sparse'] not in (0, 1)):
        raise ValueError('invalid evolution dimensions/schedule')
    if len(rows) != meta['samples'] or [r['generation'] for r in rows] != schedule(meta['steps'], meta['every']):
        raise ValueError('missing, duplicated or out-of-order evolution sample')
    first = -1
    for row in rows:
        cells, tiles = meta['cells'], meta['tiles']
        if not 0 <= row['alive'] <= row['nonzero'] <= cells or row['decay'] != row['nonzero']-row['alive']:
            raise ValueError('invalid state counters')
        if not row['changed_tiles'] <= row['changed'] <= cells or row['changed_tiles'] > tiles:
            raise ValueError('invalid change counters')
        if not row['nonzero_tiles'] <= row['influence_tiles'] <= row['work_tiles'] <= tiles:
            raise ValueError('invalid work coverage')
        if row['nonzero_tiles'] > row['nonzero'] or sum(row[k] for k in BINS) != tiles or row['empty_tiles']+row['nonzero_tiles'] != tiles:
            raise ValueError('invalid occupancy histogram')
        if row['dense'] not in (0,1) or row['scheduled_next'] > tiles:
            raise ValueError('invalid scheduler state')
        if row['fixed'] != int(row['generation'] > 0 and row['changed'] == 0):
            raise ValueError('invalid fixed point claim')
        if first >= 0 and not row['fixed']:
            raise ValueError('state changed after a claimed fixed point')
        if row['fixed'] and first < 0:
            first = row['generation']
        if meta['sparse'] and not row['dense'] and row['scheduled_next'] != row['work_tiles']:
            raise ValueError('worklist disagrees with independently scanned support/cleanup')
    if first != meta['first_observed_fixed']:
        raise ValueError('incorrect first observed fixed point')
    if not math.isclose(sum(r['scan_ms'] for r in rows), meta['scans_ms'], abs_tol=0.001):
        raise ValueError('scan time sum mismatch')
    return rows, meta


def verify_gate(path):
    text = path.read_text(encoding='utf-8')
    chunks = text.split('[EVOLVE_TEST] ')[1:]
    if len(chunks) != 20 or 'PASS 20/20 evolution final-grid checks' not in text:
        raise ValueError('incomplete evolution correctness gate')
    for chunk in chunks:
        expected = {}
        for line in chunk.splitlines():
            if line.startswith('[EVOLVE_EXPECT] '):
                values = {k:int(v) for k,v in fields(line).items()}
                expected[values['generation']] = values
        rows, _ = parse(chunk)
        for row in rows:
            if any(row[key] != value for key, value in expected[row['generation']].items()):
                raise ValueError('GPU counters differ from CPU: ' + chunk.splitlines()[0])
    print(f'PASS 20/20 CPU evolution counters and final grids: {path}')


def report(root):
    def table(path):
        with path.open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream, delimiter='\t'))
    baseline, measured = table(root/'baseline/results.tsv'), table(root/'evolution/results.tsv')
    if not baseline or [r['case'] for r in baseline] != [r['case'] for r in measured]:
        raise ValueError('baseline/evolution case list mismatch')
    samples, summaries = [], []
    for index, (base, run) in enumerate(zip(baseline, measured)):
        if any(r['verdict'] not in ('AC','TLE') or r['profile_status'] != 'ok' for r in (base,run)):
            raise ValueError('unverified output or missing phase profile')
        if base['profile_kernel'] != run['profile_kernel']:
            raise ValueError('dispatch changed in diagnostic run')
        name = Path(run['case']).stem
        rows, meta = parse((root/'evolution'/f'{index:02d}-{name}.log').read_text(encoding='utf-8'))
        for row in rows:
            row.update(case=run['case'], alive_fraction=row['alive']/meta['cells'],
                       nonzero_fraction=row['nonzero']/meta['cells'], changed_fraction=row['changed']/meta['cells'],
                       work_fraction=row['work_tiles']/meta['tiles'])
            samples.append(row)
        last = rows[-1]
        summary = dict(case=run['case'], kernel=run['profile_kernel'], steps=meta['steps'],
                       samples=meta['samples'], baseline_compute_s=base['compute_s'],
                       diagnostic_compute_s=run['compute_s'], scans_ms=meta['scans_ms'],
                       first_observed_fixed=meta['first_observed_fixed'],
                       final_alive_fraction=last['alive_fraction'], final_nonzero_fraction=last['nonzero_fraction'],
                       final_changed_fraction=last['changed_fraction'], final_work_fraction=last['work_fraction'])
        summaries.append(summary)
        print(f"{name}: base={float(base['compute_s'])*1000:.3f} ms, diagnostic={float(run['compute_s'])*1000:.3f} ms; "
              f"scan={meta['scans_ms']:.3f} ms, first_observed_fixed={meta['first_observed_fixed']}; "
              f"final nonzero={last['nonzero_fraction']:.3%}, changed={last['changed_fraction']:.3%}, work={last['work_fraction']:.3%}")
    for filename, rows in [('evolution.tsv', samples), ('summary.tsv', summaries)]:
        with (root/filename).open('x',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]),delimiter='\t')
            writer.writeheader(); writer.writerows(rows)
    print('Diagnostic timings are perturbed; do not subtract scan time to predict judge speed.')
    print('A fixed point is exact at the observed generation; -1 means none observed, not proof of no convergence/cycle.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    parser.add_argument('--gate',action='store_true')
    args = parser.parse_args()
    try:
        verify_gate(args.path) if args.gate else report(args.path)
    except (OSError,KeyError,ValueError) as error:
        print('evolution report: '+str(error),file=sys.stderr)
        sys.exit(1)
