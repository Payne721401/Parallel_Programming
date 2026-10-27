"""Synthetic diagnostic fixtures only, not measured GPU performance."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from summarize_temporal import verify, records, summarize


def record(k, steps):
    on=k>1 and steps>=k
    return dict(requested=str(k),selected=str(int(on)),depth=str(k if on else 1),
                batches=str(steps//k if on else 0),tail=str(steps%k if on else steps),
                reason='enabled' if on else 'disabled' if k==1 else 'short_run')


class TemporalTests(unittest.TestCase):
    def test_depths_and_tails(self):
        for k in (1,2,4,8):
            for steps in (1,2,3,4,7,8,9,403,1190): verify(record(k,steps),k,steps)
        wrong=record(8,403); wrong['tail']='0'
        with self.assertRaises(ValueError): verify(wrong,8,403)
        with self.assertRaises(ValueError): verify(record(1,403),2,403)

    def test_missing_or_disabled_candidate(self):
        self.assertEqual(records('[PP_PROFILE] kernel=dense_r1'),[])
        wrong=record(2,403); wrong['reason']='sparse'
        with self.assertRaises(ValueError): verify(wrong,2,403)

    def test_summary_dispatch_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for round_id,k in ((1,1),(2,2),(3,2),(4,1)):
                folder=root/f'{round_id}-{k}'; folder.mkdir()
                rows=[]
                for index,(case,steps) in enumerate((('11',403),('12',1190))):
                    rows.append(dict(case=f'cases/{case}.txt',verdict='AC',compute_s='0.2',
                                     profile_status='ok',profile_kernel='dense_r1',kernels_ms='150'))
                    (folder/f'{index:02d}-{case}.log').write_text('[PP_TEMPORAL] '+
                        ' '.join(key+'='+value for key,value in record(k,steps).items())+'\n')
                with (folder/'results.tsv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t'); w.writeheader(); w.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(root,2,True)
                with self.assertRaises(FileExistsError): summarize(root,2,True)
            (root/'2-2/00-11.log').write_text('[PP_TEMPORAL] '+
                ' '.join(key+'='+value for key,value in record(1,403).items())+'\n')
            with self.assertRaises(ValueError): summarize(root,2,True)


if __name__=='__main__': unittest.main()
