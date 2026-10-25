#!/usr/bin/env python3
"""CPU-only tests of matrix inputs/accounting; never runs CUDA or simulates timings."""
import contextlib
import csv
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import feature_matrix as matrix


class MatrixTests(unittest.TestCase):
    def test_inputs_and_comparison_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = matrix.generate(root)
            self.assertEqual(len(cases), 9)
            by_name = {c['name']: c for c in cases}
            for case in cases:
                values = matrix.describe(root / 'inputs' / (case['name'] + '.txt'))
                self.assertEqual(values[1:3], [case['width'], case['height']])
            short, long = by_name['r1_short'], by_name['r1_long']
            self.assertEqual(short['rectangles'], long['rectangles'])
            self.assertEqual(short['rule'], long['rule'])
            self.assertNotEqual(short['steps'], long['steps'])
            self.assertEqual(long['rectangles'], by_name['r1_c4']['rectangles'])
            self.assertGreater(by_name['r1_large']['width'], long['width'])
            with self.assertRaises(FileExistsError):
                matrix.generate(root)  # Never overwrite another matrix's inputs.

    def test_cluster_and_spread_are_same_count_fixed_points(self):
        cases = {c['name']: c for c in matrix.definitions()}
        counts = []
        for name in ('r1_cluster', 'r1_spread'):
            c = cases[name]
            live = {(x+dx, y+dy) for x,y,w,h,p,seed in c['rectangles']
                    for dy in range(h) for dx in range(w)}
            self.assertEqual(len(live), 1024)
            counts.append(len(live))
            neighbours = {}
            for x,y in live:
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if dx or dy:
                            q = ((x+dx) % c['width'], (y+dy) % c['height'])
                            neighbours[q] = neighbours.get(q,0)+1
            after = {q for q,n in neighbours.items() if n == 3 or (n == 2 and q in live)}
            self.assertEqual(after, live)
        self.assertEqual(counts[0], counts[1])

    def test_environment_and_slurm_guard(self):
        with patch.dict(os.environ, {'PP_KERNEL':'naive', 'PP_MOORE':'sat', 'PP_PROFILE':'0', 'PP_GRAPH':'1'}):
            env = matrix.environment()
            self.assertNotIn('PP_KERNEL', env)
            self.assertNotIn('PP_GRAPH', env)
            self.assertNotIn('PP_MOORE', env)
            self.assertEqual(env['PP_PROFILE'], '1')
            ref = matrix.environment(reference=True)
            self.assertEqual((ref['PP_KERNEL'], ref['PP_MOORE'], ref['PP_ACTIVE']), ('tiled','sat','0'))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'requires srun'):
                matrix.run(Path('must-not-be-created'))

    def test_summary_accounting_and_mismatched_activity(self):
        # Synthetic fixtures only: these are not benchmark results.
        rows = []
        for sample in (1,2):
            row = dict(name='r1_cluster', schedule='sparse', round=sample,
                       profile_kernel='sparse_r1', compute_s='0.2')
            row.update({key:'1' for key in matrix.PROFILE_TIMES})
            row.update(sparse_status='ok', sparse_selected='1', sparse_tiles_total='2048',
                       sparse_initial_work='24', sparse_mean_work='24',
                       sparse_max_work='24', sparse_dense_steps='0')
            rows.append(row)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder)
            matrix.summarize(rows,root)
            with (root/'summary.tsv').open(newline='',encoding='utf-8') as stream:
                out=list(csv.DictReader(stream,delimiter='\t'))[0]
            self.assertAlmostEqual(float(out['scheduled_tile_fraction']),24/2048)
            self.assertAlmostEqual(float(out['kernel_interval_ms_per_step']),1/256)
            rows[1]['sparse_mean_work']='25'
            with self.assertRaisesRegex(ValueError,'inconsistent activity'):
                matrix.summarize(rows,root)


if __name__ == '__main__':
    unittest.main()
