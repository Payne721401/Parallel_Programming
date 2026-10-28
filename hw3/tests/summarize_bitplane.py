#!/usr/bin/env python3
"""Validate dispatch and summarize symmetric byte/bits1/bits4 comparisons."""
import csv
import math
import re
from pathlib import Path
import statistics
import sys

METHODS = ('byte', 'bits1', 'bits4', 'bits4', 'bits1', 'byte')
def records(text, tag):
    return [dict(token.split('=',1) for token in line.split()[1:])
            for line in text.splitlines() if line.startswith('['+tag+'] ')]

def verify(record, requested, rows, selected, reason=None):
    if (record['requested'],record['rows'],record['selected']) != (str(requested),str(rows),str(selected)):
        raise ValueError('wrong bit-plane dispatch: '+str(record))
    if reason is not None and record['reason'] != reason:
        raise ValueError('wrong fallback reason: '+str(record))
    for key in ('pack_ms','evolve_ms','unpack_ms'):
        value=float(record[key])
        if not math.isfinite(value) or value<0 or (not selected and value!=0):
            raise ValueError('invalid bit-plane interval')

def verify_public_dispatch(record, row, header, mode, rows, logic):
    """Public input contract with kernel/active overrides cleared and graph/evolve off."""
    width,height=map(int,header[0].split()); steps=int(header[3])
    requested=int(width*height>=2**28 and steps>=128) if mode=='auto' else int(mode)
    radius=int(re.search(r'R(\d+)',header[1]).group(1))
    states=int(re.search(r'C(\d+)',header[1]).group(1))
    sparse=row['sparse_selected']=='1'
    selected=int(requested and steps>0 and radius==1 and states==2 and not sparse)
    if record.get('mode')!=mode or record.get('logic')!=logic:
        raise ValueError('wrong/missing mode or comparator; rebuild binaries')
    if steps==0: reason='zero_steps' if mode=='auto' or requested else 'disabled'
    elif not requested: reason='auto_small_or_short' if mode=='auto' else 'disabled'
    else: reason='not_packed_r1' if radius!=1 else 'states' if states!=2 else 'sparse' if sparse else 'enabled'
    verify(record,requested,rows,selected,reason)
    if row['profile_status']!='ok' or (row['profile_kernel']=='dense_bits') != bool(selected):
        raise ValueError('inconsistent kernel dispatch')

def gate(path, rows, kind, logic=None):
    text=path.read_text(encoding='utf-8')
    expected={'dense':452,'sparse':30,'graphs':38,'tiled':22}[kind]
    if f'PASS {expected}/{expected} ' not in text:
        raise ValueError('incomplete CPU reference gate: '+kind)
    data=records(text,'PP_BITPLANE'); profiles=records(text,'PP_PROFILE')
    if len(data)!=expected or len(profiles)!=expected:
        raise ValueError('missing gate dispatch records; rebuild binaries')
    for i,(record,profile) in enumerate(zip(data,profiles)):
        if logic is not None and record.get('logic') != logic:
            raise ValueError('wrong/missing comparator dispatch; rebuild binaries')
        if kind=='dense':
            selected=int(i<440 and i%5!=0)
            reason='enabled' if selected else 'zero_steps' if i<440 else ('states' if (i-440)%4<2 else 'not_packed_r1')
            verify(record,1,rows,selected,reason)
            if profile['kernel'] != ('dense_bits' if selected else 'none' if i<440 else 'dense_r1' if (i-440)%4<2 else 'naive'):
                raise ValueError('wrong dense gate kernel')
        else:
            verify(record,1,rows,0)
            if profile['kernel']=='dense_bits': raise ValueError('fallback entered bit kernel')
            if kind=='graphs' and record['reason']!=('zero_steps' if profile['steps']=='0' else 'graph'):
                raise ValueError('graph fallback missing')
    print(f'PASS {expected}/{expected}: {kind} CPU reference and dispatch, rows={rows}')

def summarize(root, profiled):
    samples={}
    for number,method in enumerate(METHODS,1):
        folder=root/f'{number}-{method}'
        with (folder/'results.tsv').open(encoding='utf-8',newline='') as f:
            data=list(csv.DictReader(f,delimiter='\t'))
        if len(data)!=3 or {Path(r['case']).stem for r in data}!={'01','11','12'}:
            raise ValueError('expected exactly cases 01/11/12')
        for i,row in enumerate(data):
            case=Path(row['case']).stem
            if row['verdict'] not in ('AC','TLE'): raise ValueError('output not checked correct')
            value=float(row['compute_s'])
            if not math.isfinite(value) or value<=0: raise ValueError('invalid Compute time')
            log=(folder/f'{i:02d}-{case}.log').read_text(encoding='utf-8')
            bits=records(log,'PP_BITPLANE')
            if profiled:
                if len(bits)!=1: raise ValueError('missing/duplicate dispatch')
                selected=int(method!='byte' and case!='12')
                verify(bits[0],int(method!='byte'),1 if method=='bits1' else 4,selected,
                       'disabled' if method=='byte' else 'states' if case=='12' else 'enabled')
                if row['profile_status']!='ok' or row['profile_kernel']!=('dense_bits' if selected else 'dense_r1'):
                    raise ValueError('wrong measured kernel')
                row.update(bits[0])
            elif bits or row['profile_status']!='absent':
                raise ValueError('profiling unexpectedly enabled')
            samples.setdefault((case,method),[]).append(row)
    output=[]
    for case in ('01','11','12'):
        baseline=statistics.mean(float(r['compute_s']) for r in samples[case,'byte'])
        for method in ('byte','bits1','bits4'):
            group=samples[case,method]
            values=[float(r['compute_s']) for r in group]
            mean=statistics.mean(values)
            result=dict(case=case,method=method,mean_s=mean,min_s=min(values),max_s=max(values),speedup=baseline/mean)
            if profiled:
                for key in ('kernels_ms','alloc_host_ms','h2d_ms','d2h_ms','free_host_ms','pack_ms','evolve_ms','unpack_ms'):
                    nums=[float(r[key]) for r in group]
                    if any(not math.isfinite(n) or n<0 for n in nums): raise ValueError('invalid phase')
                    result[key]=statistics.mean(nums)
            print(f'{case} {method}: {mean:.6f}s [{min(values):.6f}, {max(values):.6f}], byte/method={baseline/mean:.3f}x')
            if profiled:
                print('  kernel interval={kernels_ms:.3f} ms; pack/evolve/unpack={pack_ms:.3f}/{evolve_ms:.3f}/{unpack_ms:.3f}; free={free_host_ms:.3f}'.format(**result))
            output.append(result)
    with (root/'summary.tsv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]),delimiter='\t'); writer.writeheader(); writer.writerows(output)
    print('Two samples/method; keep all samples/outliers. Conversion costs included. No default changed.')

if __name__=='__main__':
    try:
        if len(sys.argv)==5 and sys.argv[1]=='gate': gate(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
        elif len(sys.argv)==4 and sys.argv[1]=='compare' and sys.argv[3] in ('0','1'):
            summarize(Path(sys.argv[2]),sys.argv[3]=='1')
        else: raise ValueError('usage: summarize_bitplane.py gate LOG ROWS KIND | compare DIR PROFILE')
    except (OSError,ValueError,KeyError) as error:
        print('bit-plane summary: '+str(error),file=sys.stderr); sys.exit(1)
