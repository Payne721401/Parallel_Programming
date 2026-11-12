"""CPU-only verification of overlap math and trace-coverage checks."""
import sqlite3
import tempfile
from pathlib import Path
import unittest

from nsys_summary import analyze, interval_stats


class TimelineTests(unittest.TestCase):
    def test_same_process_overlap_does_not_count_as_cross_process(self):
        self.assertEqual(interval_stats([(0, 10, 1), (5, 15, 1)]), (15, 0))

    def test_cross_process_overlap_and_touching_intervals(self):
        self.assertEqual(interval_stats([(0, 10, 1), (5, 15, 2)]), (15, 5))
        self.assertEqual(interval_stats([(0, 10, 1), (10, 20, 2)]), (20, 0))

    def test_schema_window_and_worker_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.sqlite'
            with sqlite3.connect(path) as db:
                db.executescript('''
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                INSERT INTO StringIds VALUES(1, 'gemm<float>'), (2, 'elementwise');
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER,
                    globalPid INTEGER, deviceId INTEGER, contextId INTEGER, demangledName INTEGER);
                CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT);
                ''')
                for pid, start, end in [(101, 0, 10), (102, 5, 15)]:
                    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)',
                               (start, end, pid << 24, 0, pid, 1))
                    db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)',
                               (2, 12, (pid << 24) + 1, f'diag_active2_rank{pid}'))
                # Warmup must not count toward the NVTX-window kernel totals.
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(-20,-10,?,0,101,2)', (101 << 24,))
            db.close()
            report = analyze(path, 2)
            self.assertTrue(report['all_expected_workers_observed'])
            self.assertEqual(report['devices'][0]['pids'], [101, 102])
            self.assertEqual(report['devices'][0]['kernel_union_ms'], 10 / 1e6)
            self.assertEqual(report['devices'][0]['cross_process_overlap_ms'], 5 / 1e6)
            self.assertEqual(report['gemm_targets'][0]['count'], 2)
            self.assertEqual(len(report['top_kernels']), 1)
            self.assertFalse(analyze(path, 8)['all_expected_workers_observed'])


if __name__ == '__main__':
    unittest.main()
