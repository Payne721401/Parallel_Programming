"""Execute the live source's startup policy and dispatch defaults without CUDA."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class DefaultDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        root = Path(__file__).resolve().parents[1]
        live = (root / "hw3-1.cu").read_text(encoding="utf-8")
        start = live.index("struct EagerModuleLoading {")
        end = live.index("static EagerModuleLoading eager_module_loading;", start)
        startup = live[start:end + len("static EagerModuleLoading eager_module_loading;")]
        classes = []
        for name in ("GlobalExperiments", "SparseExperiments"):
            begin = live.index("struct " + name + " {")
            classes.append(live[begin:live.index("\n};", begin) + 4])
        cls.temp = tempfile.TemporaryDirectory(prefix="hw3-live-defaults-")
        cls.addClassCleanup(cls.temp.cleanup)
        folder = Path(cls.temp.name)
        source = folder / "defaults.cpp"
        # Windows has no POSIX setenv. Only the test harness adapts that API;
        # the initializer and both policy classes are taken from live source.
        source.write_text(r'''
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ca.hpp"
#include "tile_geometry.hpp"
#ifdef _WIN32
static int host_setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && std::getenv(name)) return 0;
    return _putenv_s(name, value);
}
#define setenv host_setenv
#endif
''' + startup + "\n" + "\n".join(classes) + r'''
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    Problem p{};
    p.width = std::atoi(argv[1]); p.height = std::atoi(argv[2]);
    p.steps = std::atoi(argv[3]); p.rule.radius = std::atoi(argv[4]);
    p.rule.states = std::atoi(argv[5]);
    GlobalExperiments global(p);
    SparseExperiments followup;
    const char *loading = std::getenv("CUDA_MODULE_LOADING");
    std::printf("%s %d %d %d %d %d %d\n", loading ? loading : "unset",
        int(global.multistate_requested), int(global.rolling_requested),
        int(global.sparse_bits_requested), int(followup.warp_requested),
        int(followup.wide_requested), int(followup.output_requested));
}
''', encoding="utf-8")
        cls.binary = folder / ("defaults.exe" if os.name == "nt" else "defaults")
        subprocess.run([compiler, "-std=c++17", "-O2", "-static-libstdc++", "-static-libgcc",
                        "-I", str(root), str(source), "-o", str(cls.binary)],
                       check=True, capture_output=True, text=True, timeout=30)

    def policy(self, problem, overrides=None):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("PP_") and key != "CUDA_MODULE_LOADING"}
        env.update(overrides or {})
        return subprocess.run([str(self.binary), *map(str, problem)], env=env,
                              capture_output=True, text=True, timeout=5)

    def assert_policy(self, problem, expected, overrides=None, loading="EAGER"):
        result = self.policy(problem, overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        values = result.stdout.split()
        self.assertEqual(values[0], loading)
        self.assertEqual(tuple(map(int, values[1:])), (*expected, 0, 0, 0))

    def test_small_or_short_r1_retains_original_defaults(self):
        for states in (2, 3, 4, 8, 16):
            for problem in ((1500, 2500, 150, 1, states),
                            (8192, 8192, 1190, 1, states),
                            (16384, 8192, 127, 1, states)):
                with self.subTest(problem=problem):
                    self.assert_policy(problem, (0, 0, 0))

    def test_multistate_size_and_state_guards(self):
        for states in (3, 4, 5, 8, 16):
            for width, height in ((16384, 8192), (8192, 16384)):
                self.assert_policy((width, height, 128, 1, states), (1, 0, 0))
        self.assert_policy((16384, 8192, 128, 1, 2), (0, 0, 0))
        self.assert_policy((16383, 8192, 128, 1, 4), (0, 0, 0))

    def test_rolling_and_sparse_bits_amortization_guards(self):
        self.assert_policy((1024, 1024, 32, 2, 5), (0, 1, 0))
        self.assert_policy((1023, 1024, 32, 2, 5), (0, 0, 0))
        self.assert_policy((1024, 1024, 31, 2, 5), (0, 0, 0))
        self.assert_policy((1500, 2500, 150, 2, 5), (0, 1, 0))
        self.assert_policy((14300, 14400, 100, 5, 2), (0, 1, 0))
        self.assert_policy((17800, 13100, 50, 12, 4), (0, 1, 0))
        self.assert_policy((15300, 15350, 40, 20, 2), (0, 1, 0))
        self.assert_policy((32768, 32768, 2000, 5, 2), (0, 1, 0))
        self.assert_policy((16384, 16384, 128, 1, 2), (0, 0, 1))
        self.assert_policy((16383, 16384, 128, 1, 2), (0, 0, 0))
        self.assert_policy((16384, 16384, 127, 1, 2), (0, 0, 0))
        self.assert_policy((32768, 32768, 654, 1, 3), (1, 0, 1))

    def test_explicit_diagnostic_overrides_still_force_or_disable(self):
        flags = ("PP_MULTISTATE", "PP_ROLLING", "PP_SPARSE_BITS")
        self.assert_policy((65, 65, 7, 1, 4), (1, 1, 1), dict.fromkeys(flags, "1"))
        self.assert_policy((32768, 32768, 2000, 1, 4), (0, 0, 0),
                           dict.fromkeys(flags, "0"))
        for flag in flags:
            result = self.policy((65, 65, 7, 1, 4), {flag: "auto"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(flag + " must be 0 or 1", result.stderr)

    def test_eager_is_established_before_main_without_overriding_existing_value(self):
        self.assert_policy((65, 65, 7, 1, 4), (0, 0, 0))
        self.assert_policy((65, 65, 7, 1, 4), (0, 0, 0),
                           {"CUDA_MODULE_LOADING": "LAZY"}, loading="LAZY")


if __name__ == "__main__":
    unittest.main()
