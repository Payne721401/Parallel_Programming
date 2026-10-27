#!/usr/bin/env python3
"""Verify actual depth and batches/tails before reporting a temporal comparison."""
import csv
import math
from pathlib import Path
import statistics
import sys


def records(text):
    return [dict(t.split('=',1) for t in line.split()[1:]) for line in text.splitlines()
            if line.startswith('[PP_TEMPORAL] ')]


def verify(record, requested, steps):
    selected = requested > 1 and steps >= requested
    expected = dict(requested=requested,selected=int(selected),depth=requested if selected else 1,
                    batches=steps//requested if selected else 0,tail=steps%requested if selected else steps)
    if any(int(record[key]) != value for key,value in expected.items()):
        raise ValueError('unexpected temporal depth/batches/tail: '+str(record))
    reason = 'enabled' if selected else 'disabled' if requested==1 else 'short_run'
    if record['reason'] != reason: raise ValueError('unexpected temporal fallback')


def gate(path, depth):
    text=path.read_text(encoding='utf-8')
    if 'PASS 168/168 ' not in text: raise ValueError('incomplete temporal CPU gate')
    profiles=[dict(t.split('=',1) for t in line.split()[1:]) for line in text.splitlines()
              if line.startswith('[PP_PROFILE] ')]
    nonzero=[r for r in profiles if int(r['steps'])>0]
    data=records(text)
    if len(profiles)!=168 or len(data)!=len(nonzero): raise ValueError('missing temporal diagnostics')
    for r,p in zip(data,nonzero): verify(r,depth,int(p['steps']))
    if not any(r['selected']=='1' for r in data): raise ValueError('candidate was never executed')
    print(f'PASS 168/168 CPU comparisons; depth {depth} selection and tails verified')


def summarize(root,depth,profiled):
    groups={}
    for round_id,k in ((1,1),(2,depth),(3,depth),(4,1)):
        folder=root/f'{round_id}-{k}'
        with (folder/'results.tsv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f,delimiter='\t'))
        if len(rows)!=2 or {Path(r['case']).stem for r in rows}!={'11','12'}:
            raise ValueError('expected exactly public 11/12')
        for index,row in enumerate(rows):
            case=Path(row['case']).stem
            if row['verdict'] not in ('AC','TLE'): raise ValueError('incorrect output')
            t=float(row['compute_s'])
            if not math.isfinite(t) or t<=0: raise ValueError('invalid Compute')
            if profiled:
                if row['profile_status']!='ok' or row['profile_kernel']!='dense_r1':
                    raise ValueError('unexpected kernel path')
                data=records((folder/f'{index:02d}-{case}.log').read_text(encoding='utf-8'))
                if len(data)!=1: raise ValueError('missing/duplicate temporal record')
                verify(data[0],k,403 if case=='11' else 1190)
                if not math.isfinite(float(row['kernels_ms'])) or float(row['kernels_ms'])<0:
                    raise ValueError('invalid kernel interval')
            groups.setdefault((case,k),[]).append(row)
    output=[]
    for case in ('11','12'):
        a,b=groups[case,1],groups[case,depth]
        ta,tb=[[float(r['compute_s']) for r in rows] for rows in (a,b)]
        ma,mb=statistics.mean(ta),statistics.mean(tb)
        row=dict(case=case,depth=depth,baseline_mean_s=ma,temporal_mean_s=mb,speedup=ma/mb,
                 baseline_min_s=min(ta),baseline_max_s=max(ta),temporal_min_s=min(tb),temporal_max_s=max(tb))
        print(f'{case}: 1 -> {depth} generations: {ma:.6f} -> {mb:.6f}s; {ma/mb:.3f}x; '
              f'ranges {min(ta):.6f}..{max(ta):.6f} / {min(tb):.6f}..{max(tb):.6f}')
        if profiled:
            row['baseline_kernels_ms']=statistics.mean(float(r['kernels_ms']) for r in a)
            row['temporal_kernels_ms']=statistics.mean(float(r['kernels_ms']) for r in b)
            print(f"  kernel interval {row['baseline_kernels_ms']:.3f} -> {row['temporal_kernels_ms']:.3f} ms")
        output.append(row)
    with (root/'summary.tsv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]),delimiter='\t'); w.writeheader(); w.writerows(output)
    print('Two samples/method, same allocation. No temporal default changed.')


if __name__=='__main__':
    try:
        if len(sys.argv)<4 or sys.argv[3] not in ('2','4','8'):
            raise ValueError('usage: summarize_temporal.py gate LOG DEPTH | compare DIR DEPTH PROFILE')
        if sys.argv[1]=='gate' and len(sys.argv)==4: gate(Path(sys.argv[2]),int(sys.argv[3]))
        elif sys.argv[1]=='compare' and len(sys.argv)==5 and sys.argv[4] in ('0','1'):
            summarize(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4]=='1')
        else: raise ValueError('invalid arguments')
    except (OSError,ValueError,KeyError) as error:
        print('temporal summary: '+str(error),file=sys.stderr); sys.exit(1)
