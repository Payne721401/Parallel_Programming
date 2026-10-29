from pathlib import Path
import argparse
import os
import unittest
from unittest.mock import patch

from prepare_submission import compile_command
from verify_submission import candidate_ready, judge


class SubmissionWorkflowTests(unittest.TestCase):
    def test_official_build_uses_official_main_and_only_judge_flags(self):
        command = compile_command('nvcc', Path('/official'), Path('/release'), Path('/binary'))
        self.assertEqual(command, ['nvcc', '-std=c++17', '-O3', '-arch=sm_90', '-I'+str(Path('/official')),
                                   str(Path('/official/main.cu')), str(Path('/release/hw3-1.cu')), '-o', str(Path('/binary'))])
        gate = compile_command('nvcc', Path('/official'), Path('/release'), Path('/binary'), Path('/test.cu'))
        self.assertIn('-I'+str(Path('/release')), gate)
        self.assertNotIn(str(Path('/official/main.cu')), gate)
        self.assertNotIn('-ccbin', gate)

    def test_readiness_requires_every_default_case_and_all_three03_repeats(self):
        rows = [dict(role='candidate', case_id=f'{i:02d}', repeat=0, verdict='AC') for i in range(1, 13)]
        rows += [dict(role='candidate', case_id='03', repeat=i, verdict='AC') for i in (1, 2, 3)]
        self.assertTrue(candidate_ready(rows))
        self.assertFalse(candidate_ready(rows[:-1]))
        changed = [dict(row) for row in rows]
        changed[2]['verdict'] = 'TLE'
        self.assertFalse(candidate_ready(changed))
        changed[2]['verdict'] = 'WA'
        self.assertFalse(candidate_ready(changed))
        self.assertFalse(candidate_ready(rows+[dict(rows[0])]))
        self.assertTrue(candidate_ready(rows+[dict(role='fallback', case_id='03', repeat=0, verdict='TLE')]))

    def test_official_single_case_uses_selected_verified_source_without_inherited_flags(self):
        args = argparse.Namespace(bundle_dir=Path('/bundle'), judge_case='03', judge_role='candidate')
        manifest = dict(sources=dict(candidate=dict(directory='/candidate/source', files={'hw3-1.cu':'digest'})))
        with patch.dict(os.environ, {'PATH':'cluster', 'PP_MULTISTATE':'1', 'PP_ARENA':'1'}, clear=True), \
                patch('verify_submission.verify_prepared', return_value=manifest), \
                patch('verify_submission.shutil.which', return_value='/judge'), \
                patch('verify_submission.subprocess.run') as run, patch('builtins.print'):
            run.return_value.returncode = 0
            self.assertEqual(judge(args), 0)
            self.assertEqual(run.call_args.args[0], ['/judge', '03'])
            self.assertEqual(run.call_args.kwargs['cwd'], Path('/candidate/source'))
            self.assertFalse(any(key.startswith('PP_') for key in run.call_args.kwargs['env']))
        with patch.dict(os.environ, SLURM_JOB_ID='existing'), self.assertRaisesRegex(ValueError, 'without srun'):
            judge(args)


if __name__ == '__main__':
    unittest.main()
