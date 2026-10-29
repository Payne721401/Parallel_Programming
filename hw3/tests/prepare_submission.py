#!/usr/bin/env python3
"""Create isolated candidate/fallback submission bundles; compile, never run GPU."""
import argparse
import difflib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from global_compare import ORIGINAL_BASELINE_REVISION, compile_flags, sha256
from release_source import BASE_REVISION, VERSION, transform_release
from submission_bundle import make_submission

ROOT = Path(__file__).resolve().parents[2]
HW3 = ROOT/'hw3'
DEFAULT_CORE = Path('/work/dogeon188/pp26/judge/hw3-1/ta/core')


def git_bytes(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(),
                                   '-c', 'core.excludesFile=', *args], cwd=ROOT, timeout=30)


def export_source(revision, destination):
    destination.mkdir(parents=True, exist_ok=False)
    archive = git_bytes('archive', revision, 'hw3')
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.parent != Path('hw3'):
                continue
            if path.name == 'ca.hpp' or not (path.name == 'hw3-1.cu' or path.suffix in ('.h', '.hpp', '.cuh')):
                continue
            (destination/path.name).write_bytes(stream.extractfile(member).read())


def compile_command(nvcc, core, source, binary, gate=None):
    argv = [str(nvcc), *compile_flags('judge', False), '-I'+str(core)]
    if gate is None:
        return [*argv, str(core/'main.cu'), str(source/'hw3-1.cu'), '-o', str(binary)]
    return [*argv, '-I'+str(source), str(gate), str(source/'hw3-1.cu'), '-o', str(binary)]


def capture(argv, log, cwd):
    print('Compiling; log: '+str(log), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    if result.returncode:
        raise ValueError('compile failed; inspect '+str(log))


def prepare(args):
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    core = args.official_core_dir.resolve()
    if not args.skip_compile and not all((core/name).is_file() for name in ('main.cu', 'ca.hpp')):
        raise ValueError('official fixed files missing; use --official-core-dir with the judge core directory')
    manifest = dict(student_id=args.student_id, generating_head=git_bytes('rev-parse', 'HEAD').decode().strip(),
                    compiler_mode='judge', official_core=str(core), compiled=False,
                    policy_version=VERSION, sources={}, binaries={}, compile_commands=[])
    (output/'working-diff.log').write_bytes(git_bytes('diff', 'HEAD', '--', 'hw3'))
    (output/'status.log').write_bytes(git_bytes('status', '--short', '--untracked-files=all', '--', 'hw3'))
    for role, revision in (('candidate', BASE_REVISION), ('fallback', ORIGINAL_BASELINE_REVISION)):
        source = output/role/'source'
        export_source(revision, source)
        original = (source/'hw3-1.cu').read_bytes()
        if role == 'candidate':
            transformed = transform_release(original)
            (source/'hw3-1.cu').write_bytes(transformed)
            (output/'release.patch').write_text(''.join(difflib.unified_diff(
                original.decode().splitlines(keepends=True), transformed.decode().splitlines(keepends=True),
                fromfile='3b88c9d/hw3-1.cu', tofile='candidate/hw3-1.cu')), encoding='utf-8')
        archive = output/role/(args.student_id+'.tar')
        sources = make_submission(source, archive, args.student_id)
        manifest['sources'][role] = dict(base_revision=revision, directory=str(source),
                                         files=sources, archive=str(archive), archive_sha256=sha256(archive))
    verification = output/'verification'
    for relative in ('tests/test_hw3-1.cu', 'original_version/hw3-1.cu'):
        destination = verification/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HW3/relative, destination)
    manifest['verification_sources'] = {str(path.relative_to(output)): sha256(path)
                                        for path in verification.rglob('*.cu')}
    if not args.skip_compile:
        manifest['official_fixed_hashes'] = {name: sha256(core/name) for name in ('main.cu', 'ca.hpp')}
        nvcc = args.nvcc or ('/work/HPC_software/LMOD/nvidia/packages/cuda-12.4/bin/nvcc'
                            if Path('/work/HPC_software/LMOD/nvidia/packages/cuda-12.4/bin/nvcc').is_file() else 'nvcc')
        with (output/'compiler-version.log').open('w', encoding='utf-8') as log:
            subprocess.run([nvcc, '--version'], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=15)
        for role in ('candidate', 'fallback'):
            source = Path(manifest['sources'][role]['directory'])
            binary = output/role/'hw3-1'
            argv = compile_command(nvcc, core, source, binary)
            manifest['compile_commands'].append(argv)
            capture(argv, output/(role+'-compile.log'), source)
            manifest['binaries'][role] = dict(path=str(binary), sha256=sha256(binary))
        gate = output/'candidate-reference-test'
        argv = compile_command(nvcc, core, Path(manifest['sources']['candidate']['directory']), gate,
                               verification/'tests/test_hw3-1.cu')
        manifest['compile_commands'].append(argv)
        capture(argv, output/'gate-compile.log', verification)
        manifest['binaries']['gate'] = dict(path=str(gate), sha256=sha256(gate))
        manifest['compiled'] = True
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('Prepared: '+str(output), flush=True)
    print('Candidate archive: '+manifest['sources']['candidate']['archive'], flush=True)
    print('Original fallback: '+manifest['sources']['fallback']['archive'], flush=True)
    print('Compiled: '+str(manifest['compiled'])+'; GPU validation has not run.', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-id', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--official-core-dir', type=Path, default=DEFAULT_CORE)
    parser.add_argument('--nvcc')
    parser.add_argument('--skip-compile', action='store_true', help='source/archive preparation only; no validation claim')
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
        parser.exit(1, 'Submission preparation failed: '+str(error)+'\n')


if __name__ == '__main__':
    main()
