import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from release_source import (BASE_REVISION, SOURCE_SHA256, VERSION,
                            restore_release, transform_release)


class ReleaseSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.original = subprocess.check_output(
            ["git", "-c", "safe.directory=" + cls.root.as_posix(), "show",
             BASE_REVISION + ":hw3/hw3-1.cu"], cwd=cls.root).replace(b"\r\n", b"\n")
        cls.generated = transform_release(cls.original)

    def test_exact_revision_hash_and_byte_for_byte_inverse(self):
        self.assertEqual(hashlib.sha256(self.original).hexdigest(), SOURCE_SHA256)
        self.assertEqual(restore_release(self.generated), self.original)
        self.assertEqual(VERSION, 1)
        self.assertNotEqual(self.generated, self.original)

    def test_crlf_is_the_only_accepted_normalization(self):
        self.assertEqual(transform_release(self.original.replace(b"\n", b"\r\n")),
                         self.generated)
        self.assertEqual(restore_release(self.generated.replace(b"\n", b"\r\n")),
                         self.original)
        with self.assertRaisesRegex(ValueError, "exact 3b88"):
            transform_release(self.original.replace(b"\n", b"\r"))

    def test_unknown_or_already_generated_source_is_rejected(self):
        for source in (self.original + b"\n// unrelated edit\n", self.generated, b""):
            with self.subTest(digest=hashlib.sha256(source).hexdigest()):
                with self.assertRaisesRegex(ValueError, "exact 3b88"):
                    transform_release(source)
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            transform_release(self.original.decode())

    def test_inverse_detects_unrelated_edits(self):
        with self.assertRaisesRegex(ValueError, "unrelated modifications"):
            restore_release(self.generated + b"\n// changed\n")
        with self.assertRaisesRegex(ValueError, "requires one anchor"):
            restore_release(self.generated.replace(b"GlobalExperiments global(p);",
                                                   b"GlobalExperiments changed(p);"))

    def test_all_kernels_dispatch_allocations_and_transfers_remain_original(self):
        # Defaults are the only changes. Everything from the original profile
        # implementation through GPU kernels is retained byte-for-byte; run()
        # is identical after reversing its one construction line.
        start = b"// Diagnostic only: no events, clock reads, or extra synchronization when off."
        original_tail = self.original[self.original.index(start):]
        generated_tail = self.generated[self.generated.index(start):]
        self.assertEqual(generated_tail.replace(b"GlobalExperiments global(p);",
                                                b"GlobalExperiments global;"), original_tail)
        includes = lambda source: set(re.findall(rb'^#include "([^\"]+)"$', source, re.M))
        self.assertEqual(includes(self.generated), includes(self.original))
        for family in (b"r1_sparse_wide", b"sparse_moore_warp", b"sparse_output",
                       b"device_storage", b"device_arena"):
            self.assertNotIn(family, self.generated)


class ReleasePolicyExecutionTests(unittest.TestCase):
    """Execute the actual generated C++ constructor, without CUDA or a GPU."""

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable for host policy execution")
        root = Path(__file__).resolve().parents[2]
        original = subprocess.check_output(
            ["git", "-c", "safe.directory=" + root.as_posix(), "show",
             BASE_REVISION + ":hw3/hw3-1.cu"], cwd=root)
        generated = transform_release(original).decode()
        begin = generated.index("struct GlobalExperiments {")
        policy = generated[begin:generated.index("\n};", begin) + 4]
        cls.temp = tempfile.TemporaryDirectory(prefix="hw3-release-policy-")
        cls.addClassCleanup(cls.temp.cleanup)
        folder = Path(cls.temp.name)
        source = folder / "policy.cpp"
        source.write_text('#include <cstddef>\n#include <cstdio>\n#include <cstdlib>\n'
                          '#include <cstring>\n#include "ca.hpp"\n' + policy + r'''
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    Problem p{};
    p.width = std::atoi(argv[1]); p.height = std::atoi(argv[2]);
    p.steps = std::atoi(argv[3]); p.rule.radius = std::atoi(argv[4]);
    p.rule.states = std::atoi(argv[5]);
    GlobalExperiments global(p);
    std::printf("%d %d %d\n", int(global.multistate_requested),
        int(global.rolling_requested), int(global.sparse_bits_requested));
}
''', encoding="utf-8")
        cls.binary = folder / ("policy.exe" if os.name == "nt" else "policy")
        subprocess.run([compiler, "-std=c++17", "-O2", "-static-libstdc++", "-static-libgcc",
                        "-I", str(root / "hw3"), str(source), "-o", str(cls.binary)],
                       check=True, capture_output=True, text=True, timeout=30)

    def policy(self, width, height, steps, radius, states, overrides=None):
        env = {key: value for key, value in os.environ.items() if not key.startswith("PP_")}
        env.update(overrides or {})
        return subprocess.run([str(self.binary), *map(str, (width, height, steps, radius, states))],
                              env=env, capture_output=True, text=True, timeout=5)

    def assert_policy(self, problem, expected, overrides=None):
        result = self.policy(*problem, overrides=overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(tuple(map(int, result.stdout.split())), expected)

    def test_small_and_short_workloads_keep_all_original_defaults(self):
        for problem in ((1024, 1024, 100, 1, 4), (8192, 8192, 1190, 1, 4),
                        (16384, 8192, 127, 1, 4), (1024, 1024, 31, 2, 5),
                        (1023, 1024, 32, 2, 5), (32768, 32768, 0, 1, 4)):
            with self.subTest(problem=problem):
                self.assert_policy(problem, (0, 0, 0))

    def test_dense_multistate_boundary_and_state_guards(self):
        self.assert_policy((16384, 8192, 128, 1, 3), (1, 0, 0))
        self.assert_policy((16384, 8192, 128, 1, 16), (1, 0, 0))
        self.assert_policy((16384, 8192, 128, 1, 2), (0, 0, 0))
        self.assert_policy((16384, 8192, 128, 1, 17), (0, 0, 0))

    def test_rolling_and_sparse_boundary_guards(self):
        self.assert_policy((1024, 1024, 32, 2, 5), (0, 1, 0))
        self.assert_policy((16384, 16384, 128, 1, 2), (0, 0, 1))
        self.assert_policy((16384, 16384, 127, 1, 2), (0, 0, 0))
        self.assert_policy((16384, 16384, 128, 2, 4), (0, 1, 0))
        self.assert_policy((16384, 16384, 128, 1, 4), (1, 0, 1))

    def test_explicit_overrides_preserve_force_and_disable_semantics(self):
        flags = ("PP_MULTISTATE", "PP_ROLLING", "PP_SPARSE_BITS")
        self.assert_policy((65, 65, 7, 1, 4), (1, 1, 1), dict.fromkeys(flags, "1"))
        self.assert_policy((32768, 32768, 2000, 1, 4), (0, 0, 0), dict.fromkeys(flags, "0"))
        self.assert_policy((16384, 16384, 128, 1, 4), (0, 0, 1), {"PP_MULTISTATE": "0"})
        for flag in flags:
            with self.subTest(flag=flag):
                result = self.policy(65, 65, 7, 1, 4, {flag: "auto"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(flag + " must be 0 or 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
