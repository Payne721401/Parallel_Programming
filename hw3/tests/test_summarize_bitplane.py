"""Synthetic accounting/dispatch checks, never performance evidence."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from summarize_bitplane import gate, summarize, verify, METHODS
from run_cases import parse_profile

class BitplaneTests(unittest.TestCase):
    def test_dispatch_and_invalid_times(self):
        good=dict(requested='1',rows='4',selected='1',reason='enabled',pack_ms='1',evolve_ms='2',unpack_ms='3')
        verify(good,1,4,1,'enabled')
        for key,value in [('rows','1'),('selected','0'),('reason','sparse'),('evolve_ms','nan'),('pack_ms','-1')]:
            with self.assertRaises(ValueError): verify(dict(good,**{key:value}),1,4,1,'enabled')
        with self.assertRaises(ValueError): verify(dict(good,selected='0'),1,4,0)

    def test_dense_gate_checks_every_actual_selection(self):
        lines=[]
        for i in range(452):
            selected=i<440 and i%5!=0
            reason='enabled' if selected else 'zero_steps' if i<440 else 'states' if (i-440)%4<2 else 'not_packed_r1'
            kernel='dense_bits' if selected else 'none' if i<440 else 'dense_r1' if (i-440)%4<2 else 'naive'
            lines.extend([f'[PP_PROFILE] kernel={kernel}',
                          f'[PP_BITPLANE] requested=1 rows=4 selected={int(selected)} reason={reason} pack_ms=0 evolve_ms=0 unpack_ms=0'])
        lines.append('PASS 452/452 synthetic fixture')
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            path=Path(tmp)/'gate.log'; path.write_text('\n'.join(lines))
            gate(path,4,'dense')
            with self.assertRaises(ValueError): gate(path,4,'dense','compact')
            path.write_text('\n'.join(lines).replace('requested=1','requested=1 logic=compact'))
            gate(path,4,'dense','compact')
            with self.assertRaises(ValueError): gate(path,4,'dense','reference')
            path.write_text('\n'.join(lines).replace('selected=1','selected=0',1))
            with self.assertRaises(ValueError): gate(path,4,'dense')

    def test_summary_and_wrong_fallback(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            for number,method in enumerate(METHODS,1):
                folder=root/f'{number}-{method}'; folder.mkdir()
                rows=[]
                for i,case in enumerate(('01','11','12')):
                    selected=method!='byte' and case!='12'
                    row=dict(case=f'cases/{case}.txt',verdict='AC',compute_s='0.2' if not selected else '0.1',
                             profile_status='ok',profile_kernel='dense_bits' if selected else 'dense_r1')
                    row.update({k:'1' for k in ('kernels_ms','alloc_host_ms','h2d_ms','d2h_ms','free_host_ms')})
                    rows.append(row)
                    record=dict(requested=int(method!='byte'),selected=int(selected),rows=1 if method=='bits1' else 4,
                                reason='disabled' if method=='byte' else 'states' if case=='12' else 'enabled',
                                pack_ms=0,evolve_ms=0,unpack_ms=0)
                    (folder/f'{i:02d}-{case}.log').write_text('[PP_BITPLANE] '+' '.join(f'{k}={v}' for k,v in record.items()))
                with (folder/'results.tsv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t'); writer.writeheader(); writer.writerows(rows)
            summarize(root,True)
            with (root/'summary.tsv').open() as f: output=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual(len(output),9)
            self.assertEqual(output[1]['speedup'],'2.0')
            self.assertEqual(output[-1]['speedup'],'1.0')
            path=root/'2-bits1/02-12.log'
            path.write_text(path.read_text().replace('selected=0','selected=1'))
            with self.assertRaises(ValueError): summarize(root,True)

    def test_profile_accepts_dense_bits(self):
        text='[PP_PROFILE] kernel=dense_bits '+ ' '.join(k+'=1' for k in
             ('alloc_host_ms','h2d_ms','kernels_ms','d2h_ms','free_host_ms','run_host_ms'))
        self.assertEqual(parse_profile(text)[:2],['ok','dense_bits'])

if __name__=='__main__': unittest.main()
