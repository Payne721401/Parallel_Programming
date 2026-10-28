#!/usr/bin/env python3
"""Check factorial experiment dispatch, outputs, and paired comparator ratios."""
import csv
import math
from pathlib import Path
import statistics
import sys
from summarize_bitplane import gate, records, verify

CONFIGS = ('reference1','reference2','reference4','compact1','compact2','compact4')
METHODS = CONFIGS + CONFIGS[::-1]
CASES = ('01','11','12')
PHASES = ('kernels_ms','alloc_host_ms','h2d_ms','d2h_ms','free_host_ms',
          'pack_ms','evolve_ms','unpack_ms')

def checked_number(value, positive=False):
    number=float(value)
    if not math.isfinite(number) or number<0 or (positive and number==0):
        raise ValueError('invalid timing: '+str(value))
    return number

def summarize(root, profiled):
    samples={}
    for index,method in enumerate(METHODS,1):
        logic,rows=method[:-1],int(method[-1])
        folder=root/f'{index}-{method}'
        with (folder/'results.tsv').open(encoding='utf-8',newline='') as f:
            data=list(csv.DictReader(f,delimiter='\t'))
        if len(data)!=3 or {Path(r['case']).stem for r in data}!=set(CASES):
            raise ValueError('expected exactly cases 01/11/12')
        for i,row in enumerate(data):
            case=Path(row['case']).stem
            if row['verdict'] not in ('AC','TLE'): raise ValueError('output not verified')
            checked_number(row['compute_s'],True)
            bits=records((folder/f'{i:02d}-{case}.log').read_text(encoding='utf-8'),'PP_BITPLANE')
            if profiled:
                if len(bits)!=1 or bits[0].get('logic')!=logic:
                    raise ValueError('wrong/missing comparator dispatch; rebuild binaries')
                selected=int(case!='12')
                verify(bits[0],1,rows,selected,'enabled' if selected else 'states')
                if row['profile_status']!='ok' or row['profile_kernel']!=('dense_bits' if selected else 'dense_r1'):
                    raise ValueError('wrong kernel dispatch')
                row.update(bits[0])
                for key in PHASES: checked_number(row[key])
            elif bits or row['profile_status']!='absent':
                raise ValueError('profiling unexpectedly enabled')
            samples.setdefault((case,method),[]).append(row)
    output=[]
    for case in CASES:
        means={m:statistics.mean(float(r['compute_s']) for r in samples[case,m]) for m in CONFIGS}
        noisy={m:max(float(r['compute_s']) for r in samples[case,m]) /
                 min(float(r['compute_s']) for r in samples[case,m]) > 1.20 for m in CONFIGS}
        for method in CONFIGS:
            group=samples[case,method]; values=[float(r['compute_s']) for r in group]
            result=dict(case=case,logic=method[:-1],rows=int(method[-1]),mean_s=means[method],
                        min_s=min(values),max_s=max(values),
                        reference4_over_method=means['reference4']/means[method],
                        same_rows_reference_over_method=means['reference'+method[-1]]/means[method],
                        timing_status='unstable_samples' if noisy[method] or noisy['reference4'] or noisy['reference'+method[-1]] else 'two_samples')
            if profiled:
                for key in PHASES: result[key]=statistics.mean(float(r[key]) for r in group)
            output.append(result)
            if result['timing_status']=='unstable_samples':
                print('WARNING: sample range exceeds 20% in this configuration or its reference; retain values but do not attribute the ratio to kernel speedup.')
            print(f"{case} {method}: {means[method]:.6f}s [{min(values):.6f}, {max(values):.6f}], "
                  f"reference4/method={result['reference4_over_method']:.3f}x, "
                  f"same-rows reference/method={result['same_rows_reference_over_method']:.3f}x")
            if profiled:
                print('  kernel interval={kernels_ms:.3f} ms; pack/evolve/unpack={pack_ms:.3f}/{evolve_ms:.3f}/{unpack_ms:.3f}; alloc/free={alloc_host_ms:.3f}/{free_host_ms:.3f} ms'.format(**result))
    with (root/'summary.tsv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]),delimiter='\t')
        writer.writeheader(); writer.writerows(output)
    print('12 is unchanged fallback. Two samples/configuration; no default changed.')

if __name__=='__main__':
    try:
        if len(sys.argv)==6 and sys.argv[1]=='gate' and sys.argv[4] in ('reference','compact'):
            gate(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[5],sys.argv[4])
        elif len(sys.argv)==4 and sys.argv[1]=='compare' and sys.argv[3] in ('0','1'):
            summarize(Path(sys.argv[2]),sys.argv[3]=='1')
        else: raise ValueError('usage: summarize_bitlogic.py gate LOG ROWS LOGIC KIND | compare DIR PROFILE')
    except (OSError,ValueError,KeyError) as error:
        print('bitlogic summary: '+str(error),file=sys.stderr); sys.exit(1)
