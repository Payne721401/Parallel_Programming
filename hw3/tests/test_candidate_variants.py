import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from global_compare import (ORIGINAL_BASELINE_REVISION, comparison_binaries,
                            prepare_candidate_variant, sha256, verify_variant_methods)


class CandidateVariantTests(unittest.TestCase):
    def test_original_variant_restores_legacy_dependencies_and_keeps_current_gate_and_multistate(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            baseline, candidate, scripts = (folder/name for name in ('baseline', 'candidate', 'scripts'))
            for path in (baseline, candidate, scripts):
                path.mkdir()
            for name in ('hw3-1.cu', 'main.cu', 'ca.hpp', 'r1_bits.hpp', 'arch_stats.cuh'):
                (baseline/name).write_text('original '+name+'\n')
                (candidate/name).write_text('working '+name+'\n')
            for name in ('r1_multistate.hpp', 'r1_multistate.cuh'):
                (candidate/name).write_text('current '+name+'\n')
            (candidate/'tests').mkdir()
            (candidate/'tests/test_hw3-1.cu').write_text('current reference gate\n')
            (scripts/'lean_multistate_source.py').write_text('frozen transformer fixture\n')
            inputs = []
            def transform(source):
                inputs.append(source)
                return source+b'added multistate\n'
            transformer = SimpleNamespace(transform_original_multistate=transform, TRANSFORMER_VERSION='test')
            with patch.dict(sys.modules, lean_multistate_source=transformer), \
                    patch('global_compare.__file__', str(scripts/'global_compare.py')):
                metadata, hashes = prepare_candidate_variant('original-multistate', folder, baseline,
                                                             candidate, ORIGINAL_BASELINE_REVISION)
            self.assertEqual(inputs, [(baseline/'hw3-1.cu').read_bytes()])
            self.assertEqual((candidate/'hw3-1.cu').read_bytes(), inputs[0]+b'added multistate\n')
            for name in ('main.cu', 'ca.hpp', 'r1_bits.hpp', 'arch_stats.cuh'):
                self.assertEqual((baseline/name).read_bytes(), (candidate/name).read_bytes())
            self.assertTrue((candidate/'r1_multistate.hpp').read_text().startswith('current'))
            self.assertEqual((candidate/'tests/test_hw3-1.cu').read_text(), 'current reference gate\n')
            self.assertEqual(metadata['candidate_base_revision'], ORIGINAL_BASELINE_REVISION)
            self.assertEqual(metadata['candidate_variant'], 'original-multistate')
            self.assertEqual(set(hashes), {'source-transformer.py', 'candidate-source.patch'})
            self.assertEqual(hashes['source-transformer.py'], sha256(folder/'source-transformer.py'))
            self.assertIn('+added multistate', (folder/'candidate-source.patch').read_text())

    def test_working_variant_does_not_overlay_source_and_wrong_legacy_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            metadata, hashes = prepare_candidate_variant('working', folder, folder/'absent', folder/'absent', 'current')
            self.assertEqual(metadata, {'candidate_variant': 'working'})
            self.assertEqual(hashes, {})
            with self.assertRaisesRegex(ValueError, 'requires frozen baseline'):
                prepare_candidate_variant('original-multistate', folder, folder, folder, 'wrong')
            with self.assertRaisesRegex(ValueError, 'unknown candidate variant'):
                prepare_candidate_variant('typo', folder, folder, folder, ORIGINAL_BASELINE_REVISION)

    def test_variant_identity_survives_same_binary_role_alias(self):
        manifest = dict(baseline_revision=ORIGINAL_BASELINE_REVISION, candidate_revision='tooling-head',
                        candidate_variant='original-multistate', candidate_base_revision=ORIGINAL_BASELINE_REVISION,
                        binaries={name: dict(sha256=name) for name in ('baseline', 'candidate')})
        binaries = {name: Path(name) for name in manifest['binaries']}
        _, provenance = comparison_binaries(manifest, binaries, 'candidate')
        self.assertEqual(provenance['baseline'], provenance['candidate'])
        self.assertEqual(provenance['candidate']['source_variant'], 'original-multistate')
        self.assertEqual(provenance['candidate']['base_revision'], ORIGINAL_BASELINE_REVISION)
        self.assertEqual(provenance['candidate']['source_revision'], 'tooling-head')

    def test_lean_variant_rejects_methods_it_does_not_compile(self):
        manifest = dict(candidate_variant='original-multistate')
        for method in ('disabled', 'multistate'):
            verify_variant_methods(manifest, method, 'disabled')
        for method, baseline in (('combined', 'disabled'), ('rolling', 'disabled'), ('multistate', 'combined')):
            with self.assertRaisesRegex(ValueError, 'supports only'):
                verify_variant_methods(manifest, method, baseline)
        verify_variant_methods({}, 'combined', 'combined')


if __name__ == '__main__':
    unittest.main()
