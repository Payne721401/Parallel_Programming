"""Synthetic parser fixtures, not measured performance evidence."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from run_cases import PROFILE_COLUMNS, SPARSE_COLUMNS
from summarize_support import summarize, verify_dispatch


class SupportSummaryTests(unittest.TestCase):
    def fixture(self, root):
        for number,method in ((1,'tile'),(2,'bbox'),(3,'bbox'),(4,'tile')):
            folder=root/f'{number}-{method}'
            folder.mkdir()
            rows=[]
            for index,case in enumerate(('08','09','10')):
                row=dict(case=f'cases/{case}.txt',verdict='AC',compute_s='0.4')
                row.update({k:'1' for k in PROFILE_COLUMNS+SPARSE_COLUMNS})
                row.update(profile_status='ok',profile_kernel='sparse_sliding' if case=='09' else 'sparse_r1',
                           sparse_status='ok',sparse_tiles_total='100',sparse_initial_work='20',
                           sparse_mean_work='10' if method=='bbox' else '20',sparse_max_work='20',sparse_dense_steps='0')
                rows.append(row)
                (folder/f'{index:02d}-{case}.log').write_text(
                    f'[PP_SUPPORT] mode={method} selected={int(method=="bbox")} initial=tile\n',encoding='utf-8')
            with (folder/'results.tsv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t')
                writer.writeheader(); writer.writerows(rows)

    def test_different_work_allowed_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(root,True)
                with self.assertRaises(FileExistsError): summarize(root,True)
            self.assertTrue((root/'summary.tsv').is_file())

    def test_wrong_or_missing_selection_rejected(self):
        for text in ('','[PP_SUPPORT] mode=tile selected=0 initial=tile',
                     '[PP_SUPPORT] mode=bbox selected=0 initial=tile'):
            with self.assertRaises(ValueError): verify_dispatch(text,'bbox')

    def test_non_deterministic_work_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            path=root/'3-bbox/results.tsv'
            text=path.read_text(encoding='utf-8').replace('\t10\t20\t0','\t11\t20\t0')
            path.write_text(text,encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'non-deterministic'): summarize(root,True)


if __name__=='__main__': unittest.main()
