"""Synthetic fixtures test experiment accounting, not GPU performance."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from summarize_bitlogic import METHODS, PHASES, summarize

def fixture(root, profiled):
    for index,method in enumerate(METHODS,1):
        folder=root/f'{index}-{method}'; folder.mkdir()
        data=[]
        for i,case in enumerate(('01','11','12')):
            selected=case!='12'
            row=dict(case=f'cases/{case}.txt',verdict='AC',
                     compute_s='0.1' if selected and method.startswith('compact') else '0.2',
                     profile_status='ok' if profiled else 'absent',
                     profile_kernel='dense_bits' if selected else 'dense_r1')
            row.update({key:'1' for key in PHASES})
            data.append(row)
            record=dict(requested=1,selected=int(selected),rows=method[-1],logic=method[:-1],
                        reason='enabled' if selected else 'states',pack_ms=0,evolve_ms=0,unpack_ms=0)
            text='[PP_BITPLANE] '+' '.join(f'{k}={v}' for k,v in record.items()) if profiled else ''
            (folder/f'{i:02d}-{case}.log').write_text(text)
        with (folder/'results.tsv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),delimiter='\t')
            writer.writeheader(); writer.writerows(data)

class BitlogicTests(unittest.TestCase):
    def test_summary_both_modes_and_fallback(self):
        for profiled in (False,True):
            with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                root=Path(tmp); fixture(root,profiled); summarize(root,profiled)
                with (root/'summary.tsv').open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
                self.assertEqual(len(rows),18)
                self.assertEqual(rows[5]['same_rows_reference_over_method'],'2.0')
                self.assertEqual(rows[-1]['same_rows_reference_over_method'],'1.0')

    def test_reject_wrong_logic_rows_and_selection(self):
        for before,after in [('logic=compact','logic=reference'),('rows=4','rows=2'),
                             ('selected=1','selected=0'),('logic=compact','old_logic=compact')]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); fixture(root,True)
                path=root/'6-compact4/00-01.log'
                path.write_text(path.read_text().replace(before,after))
                with self.assertRaises(ValueError): summarize(root,True)

    def test_reject_missing_round_and_invalid_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root,False)
            path=root/'12-reference1/results.tsv'
            path.write_text(path.read_text().replace('0.2','nan',1))
            with self.assertRaises(ValueError): summarize(root,False)
            path.unlink()
            with self.assertRaises(OSError): summarize(root,False)

    def test_outlier_taints_both_sides_of_ratio_without_deleting_sample(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()) as captured:
            root=Path(tmp); fixture(root,False)
            path=root/'11-reference2/results.tsv'
            path.write_text(path.read_text().replace('0.2','10.0',1))
            summarize(root,False)
            with (root/'summary.tsv').open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual(rows[1]['timing_status'],'unstable_samples')
            self.assertEqual(rows[4]['timing_status'],'unstable_samples')
            self.assertEqual(rows[1]['max_s'],'10.0')
            self.assertIn('WARNING:',captured.getvalue())

if __name__=='__main__': unittest.main()
