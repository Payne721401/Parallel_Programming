"""Check measurement accounting and parser failures without a GPU."""
import contextlib
import csv
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import arch_measure as am


REPORT = """
  void <unnamed>::step_sparse<(bool)1>(args) (528, 1, 1)x(32, 8, 1), Context 1, Stream 7, Device 0
    Section: GPU Speed Of Light Throughput
    Metric Name                  Metric Unit    Metric Value
    --------------------------   -----------    ------------
    Duration                     msecond        1.83
    Compute (SM) Throughput       %              92.91
    DRAM Throughput               %              6.52

    Section: Scheduler Statistics
    Metric Name                  Metric Unit    Metric Value
    No Eligible                  %              6.85
    Eligible Warps Per Scheduler  warp           5.19

    Section: Launch Statistics
    Metric Name                  Metric Unit    Metric Value
    Registers Per Thread         register/thread  32

    Section: Occupancy
    Metric Name                  Metric Unit    Metric Value
    Achieved Occupancy           %              96.91

    WRN   optional sampling metric unavailable
"""


class MeasurementTests(unittest.TestCase):
    def test_duration_units(self):
        for unit, scale in (("nsecond", .001), ("usecond", 1), ("msecond", 1000), ("second", 1000000)):
            result, metrics = am.parse_ncu(REPORT.replace("msecond", unit))
            self.assertAlmostEqual(result["duration_us"], 1.83 * scale)
            self.assertEqual(result["registers"], 32)
            self.assertEqual(len(metrics), 7)

    def test_fail_missing_duplicate_and_nonfinite(self):
        for text in (REPORT + REPORT, REPORT.replace("No Eligible", "Absent"),
                     REPORT.replace("92.91", "nan"), REPORT.replace("92.91", "-1"),
                     REPORT.replace("msecond", "unknown"), "no matching kernels"):
            with self.assertRaises(ValueError):
                am.parse_ncu(text)

    def test_phase_offsets(self):
        for steps in (40, 50, 100, 403, 654, 1190, 2000, 3640):
            phases = dict(am.windows(steps))
            self.assertEqual(phases["middle"] + 1, steps // 2 + 1)
            self.assertEqual(phases["late"] + 1, steps - 2)
            self.assertTrue(all(0 <= skip < steps for skip in phases.values()))
        with self.assertRaises(ValueError):
            am.windows(0)

    def test_capture_is_bounded_and_checked_before_summary(self):
        for checker_fails in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                previous = Path.cwd()
                try:
                    os.chdir(tmp)
                    Path("cases").mkdir()
                    Path("cases/08.txt").write_text("32 32\nrule\ntorus\n3640\n")
                    root = Path("logs"); root.mkdir()
                    calls = []

                    def fake_command(argv, log, timeout=150):
                        calls.append(argv)
                        if argv[0] == "hw3-1-check" and checker_fails:
                            raise ValueError("checker rejected output")
                        text = "\n".join(am.SECTIONS) if "--list-sections" in argv else REPORT if "--import" in argv else "ok"
                        log.write_text(text)

                    with patch.object(am, "command", side_effect=fake_command), patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                        if checker_fails:
                            with self.assertRaises(ValueError):
                                am.ncu(("08",), root)
                            self.assertFalse((root / "ncu-summary.tsv").exists())
                        else:
                            am.ncu(("08",), root)
                            with (root / "ncu-summary.tsv").open() as stream:
                                rows = list(csv.DictReader(stream, delimiter="\t"))
                            self.assertEqual([r["generation"] for r in rows], ["1821", "3638"])
                            self.assertTrue(all(r["output_verified"] == "1" for r in rows))
                            captures = [c for c in calls if "--export" in c]
                            self.assertEqual(len(captures), 2)
                            self.assertTrue(all(c[c.index("--launch-count")+1] == "1" for c in captures))
                            self.assertEqual(sum(c[0] == "hw3-1-check" for c in calls), 2)
                finally:
                    os.chdir(previous)

    def test_timing_keeps_outliers_separate_from_profile_and_rejects_wa(self):
        for verdict in ("TLE", "WA"):
            with tempfile.TemporaryDirectory() as tmp:
                previous = Path.cwd()
                try:
                    os.chdir(tmp)
                    Path("cases").mkdir()
                    Path("cases/12.txt").write_text("16384 8192\nR1,C4\ndead\n1190\n")
                    root = Path("logs"); root.mkdir()
                    values = iter(("0.1", "0.2", "0.1", "50"))

                    def fake_run(args):
                        profiled = os.environ["PP_PROFILE"] == "1"
                        folder = Path(args.outdir); folder.mkdir()
                        row = dict(case="cases/12.txt", verdict=verdict, compute_s=next(values),
                                   profile_status="ok" if profiled else "absent",
                                   profile_kernel="dense_r1" if profiled else "", sparse_selected="",
                                   sparse_mean_work="", sparse_max_work="", sparse_dense_steps="")
                        row.update({k: "1" for k in am.PROFILE_TIMES})
                        am.write_table(folder / "results.tsv", [row])
                        if profiled:
                            (folder / "00-12.log").write_text(
                                "[PP_BITPLANE] mode=auto logic=compact requested=0 selected=0 rows=4 "
                                "reason=auto_small_or_short pack_ms=0 evolve_ms=0 unpack_ms=0\n")
                        return 1

                    with patch.object(am, "run_cases", side_effect=fake_run), patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                        if verdict == "WA":
                            with self.assertRaises(ValueError):
                                am.timing(("12",), root)
                            self.assertFalse((root / "timing-summary.tsv").exists())
                        else:
                            am.timing(("12",), root)
                            with (root / "timing-summary.tsv").open() as stream:
                                row = next(csv.DictReader(stream, delimiter="\t"))
                            self.assertEqual(row["median_s"], "0.1")
                            self.assertEqual(row["max_s"], "0.2")
                            self.assertEqual(row["profile_compute_s"], "50")
                            self.assertEqual(row["timing_status"], "unstable_samples")
                            self.assertEqual(row["samples"], "3")
                finally:
                    os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
