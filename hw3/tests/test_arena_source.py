"""CPU source/provenance checks; no CUDA compiler, context or GPU is used."""

from pathlib import Path
import re
import subprocess
import unittest

from arena_source import (restore_multistate_arena, transform_multistate_arena,
                          restore_original_multistate_arena,
                          transform_original_multistate_arena)
from lean_multistate_source import (ORIGINAL_BASELINE_REVISION,
                                    restore_original_multistate,
                                    transform_original_multistate)


class ArenaSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.original = subprocess.check_output(
            ["git", "-c", "safe.directory=" + cls.root.as_posix(), "show",
             ORIGINAL_BASELINE_REVISION + ":hw3/hw3-1.cu"], cwd=cls.root)
        cls.lean = transform_original_multistate(cls.original)
        cls.generated = transform_multistate_arena(cls.lean)

    def test_two_stage_inverse_recovers_exact_original(self):
        self.assertEqual(restore_multistate_arena(self.generated), self.lean)
        self.assertEqual(restore_original_multistate(
            restore_multistate_arena(self.generated)), self.original)
        self.assertIs(transform_original_multistate_arena, transform_multistate_arena)
        self.assertIs(restore_original_multistate_arena, restore_multistate_arena)

    def test_accepts_only_verified_lean_input_and_crlf_normalization(self):
        self.assertEqual(transform_multistate_arena(self.lean.replace(b"\n", b"\r\n")),
                         self.generated)
        self.assertEqual(restore_multistate_arena(
            self.generated.replace(b"\n", b"\r\n")), self.lean)
        for source in (self.original, self.generated, b"", self.lean + b"\n// unrelated\n",
                       self.lean.replace(b"#include <vector>", b"#include <array>"),
                       self.lean.replace(b"\n", b"\r")):
            with self.subTest(source=source[:32]):
                with self.assertRaises(ValueError):
                    transform_multistate_arena(source)
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            transform_multistate_arena(self.lean.decode())

    def test_inverse_rejects_changed_generated_source(self):
        for source in (self.generated + b"\n// changed after freezing\n",
                       self.generated.replace(b"storage.release();", b"storage.release_twice();"),
                       self.generated.replace(b"#include <vector>", b"#include <array>")):
            with self.subTest(source=source[-60:]):
                with self.assertRaises(ValueError):
                    restore_multistate_arena(source)

    def test_only_host_headers_added_and_no_new_device_code(self):
        def includes(source):
            return set(re.findall(rb'^#include "([^"]+)"$', source, re.M))
        self.assertEqual(includes(self.generated) - includes(self.lean),
                         {b"device_arena.hpp", b"device_storage.cuh"})
        self.assertFalse(includes(self.lean) - includes(self.generated))
        for token in (b"__global__", b"__device__", b"<<<"):
            self.assertEqual(self.generated.count(token), self.lean.count(token))
        for filename in ("device_arena.hpp", "device_storage.cuh"):
            header = (self.root / "hw3" / filename).read_bytes()
            for token in (b"__global__", b"__device__", b"<<<"):
                self.assertNotIn(token, header)
        self.assertLess(self.generated.index(b"struct SparseWorkspace {"),
                        self.generated.index(b'#include "device_storage.cuh"'))
        self.assertLess(self.generated.index(b'#include "device_storage.cuh"'),
                        self.generated.index(b"void run(uint8_t *grid"))

    def test_kernels_workspace_and_generation_chain_unchanged(self):
        def region(source, first, last):
            begin = source.index(first)
            return source[begin:source.index(last, begin)]
        for first, last in (
            (b"__device__ __forceinline__ int wrap_neighbour", b'#include "r1_bitplane.cuh"'),
            (b"struct SparseWorkspace {", b'#include "evolution_diag.cuh"'),
            (b"    const bool sparse =", b"    const auto free_start ="),
        ):
            with self.subTest(first=first):
                expected = region(self.lean, first, last)
                actual = region(self.generated, first, last)
                if first == b"struct SparseWorkspace {":
                    actual = actual.replace(b'#include "device_storage.cuh"\n', b"")
                self.assertEqual(actual, expected)

    def test_arena_uses_existing_capacity_and_selected_workspace_geometry(self):
        # Source markers tie the generated experiment to the real host storage
        # API; padded widths must retain allocation_bytes, never just byte cells.
        self.assertIn(b"storage.allocate(allocation_bytes, inspect_sparse, workspace_tiles, profile.enabled)",
                      self.generated)
        self.assertIn(b"storage.bind_sparse(sparse_work, p, TILE_W, TILE_H)", self.generated)
        self.assertIn(b"src = storage.pointer<uint8_t>(storage.layout.src)", self.generated)
        self.assertIn(b"dst = storage.pointer<uint8_t>(storage.layout.dst)", self.generated)
        self.assertNotIn(b"r1_sparse_wide", self.generated)

    def test_owner_release_follows_all_consumers_and_original_path_survives(self):
        marker = self.generated.index(b"    if (storage.selected) {\n")
        for consumer in (b"    profile.collect();", b"    bitplane.finish();",
                         b"    evolution.finish(p, sparse);", b"cudaGraphExecDestroy(executable)",
                         b"cudaGraphDestroy(graph)", b"cudaStreamDestroy(stream)"):
            self.assertLess(self.generated.index(consumer), marker)
        ending = self.generated[marker:]
        owned_branch, old_branch = ending.split(b"    } else {\n", 1)
        self.assertIn(b"storage.release();", owned_branch)
        self.assertNotIn(b"cudaFree(", owned_branch)
        self.assertNotIn(b"sparse_work.release()", owned_branch)
        self.assertIn(b'check(cudaFree(dst), "free destination")', old_branch)
        self.assertIn(b'check(cudaFree(src), "free source")', old_branch)
        self.assertIn(b"sparse_work.release();", old_branch)
        # T=0 returns before allocate while still emitting requested/selected.
        early = self.generated.index(b"    if (p.steps == 0 && !evolution.enabled)")
        early_return = self.generated.index(b"        return;", early)
        self.assertLess(self.generated.index(b"storage.report(profile.enabled);", early), early_return)
        self.assertLess(early_return, self.generated.index(b"storage.allocate("))
        self.assertEqual(self.generated.count(b"storage.report(profile.enabled);"), 2)


if __name__ == "__main__":
    unittest.main()
