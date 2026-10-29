import hashlib
from pathlib import Path
import re
import subprocess
import unittest

from lean_multistate_source import (ORIGINAL_BASELINE_REVISION, ORIGINAL_SOURCE_SHA256,
                                    restore_original_multistate, transform_original_multistate)


class LeanMultistateSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.original = subprocess.check_output(
            ["git", "-c", "safe.directory=" + root.as_posix(), "show",
             ORIGINAL_BASELINE_REVISION + ":hw3/hw3-1.cu"], cwd=root)
        cls.generated = transform_original_multistate(cls.original)

    def test_exact_original_hash_and_byte_for_byte_inverse(self):
        self.assertEqual(hashlib.sha256(self.original).hexdigest(), ORIGINAL_SOURCE_SHA256)
        self.assertEqual(restore_original_multistate(self.generated), self.original)
        self.assertNotEqual(self.generated, self.original)

    def test_crlf_is_the_only_accepted_source_normalization(self):
        self.assertEqual(transform_original_multistate(self.original.replace(b"\n", b"\r\n")),
                         self.generated)
        with self.assertRaisesRegex(ValueError, "exact 1352"):
            transform_original_multistate(self.original.replace(b"\n", b"\r"))

    def test_unknown_or_already_transformed_source_is_rejected(self):
        for source in (self.original + b"\n// another source edit\n", self.generated,
                       self.original.replace(b"#include <vector>", b"#include <array>"), b""):
            with self.subTest(digest=hashlib.sha256(source).hexdigest()):
                with self.assertRaisesRegex(ValueError, "exact 1352"):
                    transform_original_multistate(source)
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            transform_original_multistate(self.original.decode())

    def test_inverse_detects_any_unrelated_source_edit(self):
        with self.assertRaisesRegex(ValueError, "unrelated modifications"):
            restore_original_multistate(self.generated + b"\n// changed after generation\n")
        with self.assertRaisesRegex(ValueError, "requires one anchor"):
            restore_original_multistate(self.generated.replace(b"struct LeanMultistateExperiment",
                                                               b"struct ChangedExperiment"))

    def test_only_two_new_headers_and_no_other_new_device_families(self):
        def includes(source):
            return set(re.findall(rb'^#include "([^"]+)"$', source, re.M))
        self.assertEqual(includes(self.generated) - includes(self.original),
                         {b"r1_multistate.hpp", b"r1_multistate.cuh"})
        self.assertFalse(includes(self.original) - includes(self.generated))
        for family in (b"rolling_filters", b"r1_sparse_bits", b"r1_sparse_wide",
                       b"sparse_moore_warp", b"sparse_output"):
            self.assertNotIn(family, self.generated)

    def test_original_update_and_transfer_regions_are_identical(self):
        # The experiment must not accidentally rewrite old GPU kernels, sparse
        # workspace/dispatch, or common output/free spans while adding a branch.
        def region(source, first, last):
            begin = source.index(first)
            return source[begin:source.index(last, begin)]
        for first, last in (
            (b"__device__ __forceinline__ int wrap_neighbour", b'#include "r1_bitplane.cuh"'),
            (b"struct SparseWorkspace {", b'#include "evolution_diag.cuh"'),
            (b"    profile.mark(2);\n    // Blocking D2H", b"    profile.report(sparse ?"),
        ):
            with self.subTest(first=first):
                self.assertEqual(region(self.original, first, last),
                                 region(self.generated, first, last))
        self.assertEqual(region(self.original, b"    auto launch_step =", b"    if (bitplane.selected)"),
                         region(self.generated, b"    auto launch_step =", b"    if (multistate.selected)"))


if __name__ == "__main__":
    unittest.main()
