"""CPU-only validation of diagnostic rejection and comparison accounting."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_graphs as graphs


class GraphComparisonTests(unittest.TestCase):
    def test_dispatch_replay_and_tail(self):
        text = '[PP_GRAPH] requested=1 selected=1 batch=32 replays=3 tail=1 build_host_ms=0.4'
        self.assertEqual(graphs.graph_diagnostic(text, 1, 97), 0.4)
        for bad in ('', text+'\n'+text, text.replace('selected=1', 'selected=0'),
                    text.replace('tail=1', 'tail=0'), text.replace('0.4', 'nan')):
            with self.assertRaises(ValueError):
                graphs.graph_diagnostic(bad, 1, 97)

    def test_slurm_guard(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'requires srun'):
                graphs.run(Path('missing'), Path('must-not-be-created'), 1)

    def test_summary_rejects_activity_differences(self):
        # Artificial numbers for accounting tests only, not performance results.
        rows = []
        for name, active in graphs.JOBS:
            for mode in (0, 1):
                for repeat in (0, 1):
                    row = dict(name=name, graph=str(mode), compute_s=str(0.2/(mode+1)),
                               profile_kernel='test', graph_build_ms='0.4')
                    row.update({key:'0' for key in graphs.SPARSE_COLUMNS})
                    rows.append(row)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            graphs.summarize(rows, Path(directory))
            self.assertIn('2.0', (Path(directory)/'summary.tsv').read_text())
            rows[2]['sparse_mean_work'] = '1'
            with self.assertRaisesRegex(ValueError, 'activity changed'):
                graphs.summarize(rows, Path(directory))


if __name__ == '__main__':
    unittest.main()
