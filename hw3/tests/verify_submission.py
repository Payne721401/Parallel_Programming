#!/usr/bin/env python3
"""Manual single-allocation release validation; no judge or GPU allocation."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from global_compare import (configure, gate_plan, sha256, validate_gate, verify_gate_dispatch, write_table)
from prepare_submission import HW3
from run_cases import run_cases


def verify_prepared(folder):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('compiled') or manifest.get('compiler_mode') != 'judge':
        raise ValueError('requires completed judge-mode compilation, not --skip-compile preparation')
    for role in ('candidate', 'fallback'):
        item = manifest['sources'][role]
        root = Path(item['directory'])
        for relative, digest in item['files'].items():
            if sha256(root/relative) != digest:
                raise ValueError('source changed: '+str(root/relative))
        if sha256(Path(item['archive'])) != item['archive_sha256']:
            raise ValueError('archive changed: '+item['archive'])
    for item in manifest['binaries'].values():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('compiled binary changed: '+item['path'])
    for relative, digest in manifest['verification_sources'].items():
        if sha256(folder/relative) != digest:
            raise ValueError('verification source changed: '+relative)
    core = Path(manifest['official_core'])
    for name, digest in manifest['official_fixed_hashes'].items():
        if sha256(core/name) != digest:
            raise ValueError('official fixed file changed: '+name)
    return manifest


def clear_pp():
    for key in list(os.environ):
        if key.startswith('PP_'):
            del os.environ[key]


def candidate_ready(rows):
    candidate = [row for row in rows if row['role'] == 'candidate']
    return (len(candidate) == 15 and all(row['verdict'] == 'AC' for row in candidate)
            and {row['case_id'] for row in candidate if row['repeat'] == 0} == {f'{i:02d}' for i in range(1, 13)}
            and [row['case_id'] for row in candidate if row['repeat'] != 0] == ['03']*3)


def verify(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('requires an existing srun allocation; this script does not request one')
    if not shutil.which('hw3-1-check'):
        raise ValueError('hw3-1-check is missing from PATH')
    folder = args.bundle_dir.resolve()
    manifest = verify_prepared(folder)
    logs = folder/('validation-'+os.environ['SLURM_JOB_ID']+'-'+str(os.getpid()))
    logs.mkdir(exist_ok=False)
    deadline = time.monotonic()+420
    def remaining():
        value = deadline-time.monotonic()
        if value <= 2:
            raise ValueError('420-second validation budget exhausted')
        return value
    gate = manifest['binaries']['gate']['path']
    # Default small-input path plus explicit reference coverage of the measured kernels.
    plans = [dict(option='--quick', role='automatic', required=None, zero=None, overrides={}), *gate_plan('combined')]
    gate_rows = []
    for index, plan in enumerate(plans):
        if plan['role'] == 'automatic':
            clear_pp()
            os.environ['PP_PROFILE'] = '1'
        else:
            configure('combined', plan['role'] == 'candidate', True, **plan['overrides'])
        path = logs/f'gate-{index:02d}-{plan["option"][2:]}.log'
        with path.open('w', encoding='utf-8') as stream:
            result = subprocess.run([gate, plan['option']], stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=min(120, remaining()))
        if result.returncode:
            raise ValueError('reference gate failed: '+str(path))
        text = path.read_text(encoding='utf-8', errors='replace')
        expected = {'--quick':22, '--multistate':384, '--rolling':128, '--sparse-bits':136,
                    '--sparse':30, '--support':28}[plan['option']]
        count = validate_gate(text, expected)
        if plan['role'] != 'automatic':
            env = {key:value for key,value in os.environ.items() if key.startswith('PP_')}
            verify_gate_dispatch(text, env, count, plan['required'], plan['zero'])
        gate_rows.append(dict(option=plan['option'], role=plan['role'], count=count, verdict='PASS'))
        print(f'Gate {index} {plan["option"]}: PASS {count}/{count}', flush=True)
    write_table(logs/'gates.tsv', gate_rows)
    rows = []
    schedule = [(role, f'{case:02d}', 0) for role in ('candidate', 'fallback') for case in range(1, 13)]
    schedule += [('candidate', '03', repeat) for repeat in (1, 2, 3)]
    for index, (role, case, repeat) in enumerate(schedule):
        clear_pp()  # The actual submission defaults, not feature overrides.
        case_log = logs/f'{index:02d}-{role}-{case}-repeat{repeat}'
        call = argparse.Namespace(binary=manifest['binaries'][role]['path'], checker='hw3-1-check',
                                  inputs=[str(HW3/'cases'/(case+'.txt'))], outdir=str(case_log),
                                  wall_timeout=min(20, remaining()), limit=None)
        run_cases(call)
        with (case_log/'results.tsv').open(encoding='utf-8', newline='') as stream:
            measured, = list(csv.DictReader(stream, delimiter='\t'))
        rows.append(dict(role=role, case_id=case, repeat=repeat, **measured))
        write_table(logs/'samples.tsv', rows)
        if measured['verdict'] not in ('AC', 'TLE'):
            raise ValueError('invalid output/process: '+str(case_log))
    verify_prepared(folder)
    ready = candidate_ready(rows)
    fallback_ready = all(row['verdict'] == 'AC' for row in rows if row['role'] == 'fallback')
    result = dict(candidate_ready=ready, fallback_ready=fallback_ready, logs=str(logs),
                  candidate_archive=manifest['sources']['candidate']['archive'],
                  fallback_archive=manifest['sources']['fallback']['archive'])
    (logs/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if ready:
        (logs/'CANDIDATE_READY').write_text('All reference gates, 12 default public cases and 3 extra03 runs passed.\n')
    print(json.dumps(result, indent=2), flush=True)
    print('No new speedup or hidden-case guarantee follows from this short release check.', flush=True)
    return 0 if ready else 1


def judge(args):
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('official judge creates its own allocation; run from the login node without srun')
    folder = args.bundle_dir.resolve()
    manifest = verify_prepared(folder)
    executable = shutil.which('hw3-1-judge')
    if not executable:
        raise ValueError('hw3-1-judge is not on PATH')
    source = Path(manifest['sources'][args.judge_role]['directory'])
    environment = {key:value for key,value in os.environ.items() if not key.startswith('PP_')}
    argv = [executable] + ([] if args.judge_case == 'all' else [args.judge_case])
    print('Official judge source: '+str(source), flush=True)
    print('Role: '+args.judge_role+'; all PP_* overrides removed.', flush=True)
    print('Source SHA256: '+manifest['sources'][args.judge_role]['files']['hw3-1.cu'], flush=True)
    return subprocess.run(argv, cwd=source, env=environment, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--judge-case', choices=['all', *[f'{i:02d}' for i in range(1, 13)]],
                        help='official judge from verified source on the login node, without srun')
    parser.add_argument('--judge-role', choices=['candidate', 'fallback'], default='candidate')
    args = parser.parse_args()
    try:
        return judge(args) if args.judge_case else verify(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, 'Submission validation failed: '+str(error)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
