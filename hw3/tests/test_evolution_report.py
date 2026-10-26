"""CPU-only tests of evolution evidence validation; fixtures are not measurements."""
import unittest
import contextlib
import csv
import io
from pathlib import Path
import tempfile
from evolution_report import parse, schedule, report, verify_gate


def fixture(oscillator=False):
    lines=[]
    for t in (0,1,2):
        change=4 if oscillator and t else 0
        lines.append(f'[PP_EVOLVE] generation={t} alive=4 nonzero=4 decay=0 changed={change} '
                     f'nonzero_tiles=1 changed_tiles={int(change>0)} influence_tiles=1 work_tiles=1 '
                     'empty_tiles=0 tiles_1_16=1 tiles_17_128=0 tiles_129_512=0 tiles_513_up=0 '
                     f'scheduled_next=1 dense=0 fixed={int(t>0 and not oscillator)} scan_ms=0.1 update_ms=0.2')
    lines.append('[PP_EVOLVE_META] steps=2 cells=16 tiles=1 samples=3 every=128 sparse=1 '
                 f'first_observed_fixed={-1 if oscillator else 1} setup_host_ms=0.1 scans_ms=0.3 collect_host_ms=0.1 free_host_ms=0.1')
    return '\n'.join(lines)


class EvolutionReportTests(unittest.TestCase):
    def test_cadence(self):
        self.assertEqual(schedule(0,128),[0])
        self.assertEqual(schedule(200,128),[0,1,2,4,8,16,32,64,128,200])
        self.assertEqual(len(schedule(5000,128)),48)
        self.assertEqual(schedule(5,1),list(range(6)))

    def test_fixed_and_same_population_oscillator(self):
        rows, meta=parse(fixture())
        self.assertEqual(meta['first_observed_fixed'],1)
        rows, meta=parse(fixture(True))
        self.assertEqual(meta['first_observed_fixed'],-1)
        self.assertEqual(rows[-1]['alive'],rows[0]['alive'])
        self.assertGreater(rows[-1]['changed'],0)

    def test_reject_corrupt_evidence(self):
        text=fixture()
        for bad in (text.replace('samples=3','samples=2'), text.replace('decay=0','decay=1'),
                    text.replace('scans_ms=0.3','scans_ms=nan'), text.replace('work_tiles=1','work_tiles=0'),
                    text.replace('scheduled_next=1','scheduled_next=0'), text.replace('tiles_1_16=1','tiles_1_16=0'),
                    text.replace('first_observed_fixed=1','first_observed_fixed=-1'),
                    text.replace('generation=1','generation=2')):
            with self.assertRaises(ValueError): parse(bad)

    def test_report_and_gate_round_trip(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory)
            for name,verdict in [('baseline','AC'),('evolution','TLE')]:
                folder=root/name; folder.mkdir()
                with (folder/'results.tsv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['case','verdict','profile_status','profile_kernel','compute_s'],delimiter='\t')
                    writer.writeheader()
                    writer.writerow(dict(case='cases/05.txt',verdict=verdict,profile_status='ok',profile_kernel='tiled',compute_s='1'))
            (root/'evolution/00-05.log').write_text(fixture())
            report(root)
            self.assertTrue((root/'evolution.tsv').is_file())
            self.assertIn('first_observed_fixed',(root/'summary.tsv').read_text())
            expected=[]
            for line in fixture().splitlines():
                if line.startswith('[PP_EVOLVE] '):
                    expected.append(line.split(' scheduled_next=')[0].replace('[PP_EVOLVE]','[EVOLVE_EXPECT]'))
            gate=''.join(f'[EVOLVE_TEST] id={i} steps=2\n'+ '\n'.join(expected)+'\n'+fixture()+'\n' for i in range(20))
            gate+='PASS 20/20 evolution final-grid checks\n'
            (root/'gate.log').write_text(gate)
            verify_gate(root/'gate.log')
            (root/'gate.log').write_text(gate.replace('[EVOLVE_EXPECT] generation=1 alive=4','[EVOLVE_EXPECT] generation=1 alive=3',1))
            with self.assertRaisesRegex(ValueError,'differ from CPU'):
                verify_gate(root/'gate.log')


if __name__=='__main__': unittest.main()
