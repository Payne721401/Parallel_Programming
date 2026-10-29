"""Verify official compiler flags without invoking nvcc or a GPU."""

from pathlib import Path
import unittest
from unittest.mock import patch

from global_compare import compile_flags, main


class CompilerModeTests(unittest.TestCase):
    def test_judge_flags_ignore_cluster_compiler_availability(self):
        for available in (False, True):
            self.assertEqual(compile_flags("judge", available),
                             ["-std=c++17", "-O3", "-arch=sm_90"])

    def test_cluster_mode_retains_explicit_legacy_runtime_selection(self):
        self.assertEqual(compile_flags("cluster", False), compile_flags("judge", False))
        self.assertEqual(compile_flags("cluster", True), [
            "-std=c++17", "-O3", "-arch=sm_90", "-ccbin",
            "/work/HPC_software/LMOD/gcc/12.5.0/bin/g++",
            "-Xcompiler", "-static-libstdc++,-static-libgcc"])

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown compiler mode"):
            compile_flags("arbitrary", True)

    def test_build_cli_defaults_to_judge_and_accepts_cluster(self):
        for arguments, expected in (([], "judge"), (["--compiler-mode", "cluster"], "cluster")):
            with patch("global_compare.os.chdir"), patch("global_compare.prepare_build") as build:
                main(["build", "--build-dir", "unused", *arguments])
            self.assertEqual(build.call_args.args[0].compiler_mode, expected)
            self.assertEqual(build.call_args.args[0].build_dir, Path("unused"))


if __name__ == "__main__":
    unittest.main()
