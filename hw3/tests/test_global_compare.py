import copy
import contextlib
import io
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from global_compare import (GATE_COUNTS, case_selection, compare, configure, phase_summary, profile_fields,
                            sha256, summarize, validate_gate, verify_global, verify_manifest)
from global_compare import same_fixed_source, verify_gate_dispatch
from run_cases import PROFILE_TIMES


def timing_samples():
    return [dict(case_id="10", method=method, profiled=0, verdict=verdict, compute_s=value)
            for method, verdict, value in (("baseline", "AC", ".3"), ("candidate", "AC", ".1"),
                                           ("candidate", "AC", ".11"), ("baseline", "TLE", "3"))]


class GlobalComparisonTests(unittest.TestCase):
    def test_fixed_source_guard_accepts_crlf_only_and_rejects_content_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp)/"left.cu", Path(temp)/"right.cu"
            left.write_bytes(b"fixed file\nline two\n")
            right.write_bytes(b"fixed file\r\nline two\r\n")
            self.assertTrue(same_fixed_source(left, right))
            self.assertNotEqual(sha256(left), sha256(right))
            right.write_bytes(b"changed file\r\nline two\r\n")
            self.assertFalse(same_fixed_source(left, right))
            right.write_bytes(b"fixed file\rline two\r")
            self.assertFalse(same_fixed_source(left, right))

    def test_configuration_clears_every_inherited_override(self):
        with patch.dict(os.environ, PP_KERNEL="naive", PP_ARCH="all", PP_GRAPH="1", PP_FUTURE="1"):
            env = configure("multistate", True)
            self.assertNotIn("PP_FUTURE", env)
            self.assertNotIn("PP_KERNEL", env)
            self.assertEqual(env["PP_MULTISTATE"], "1")
            self.assertEqual(env["PP_ROLLING"], "0")
            self.assertEqual(env["PP_GRAPH"], "0")
            env = configure("combined", False, True)
            self.assertEqual(env["PP_PROFILE"], "1")
            self.assertEqual(env["PP_MULTISTATE"], "0")
            self.assertEqual(env["PP_ROLLING"], "0")
            self.assertEqual(env["PP_SPARSE_BITS"], "0")

    def test_case_ranges_guard_and_invalid_selections(self):
        self.assertEqual(case_selection("04-06,06,12", "target"), ("04", "05", "06", "12", "11"))
        self.assertEqual(case_selection(None, "sparse"), ("08", "09", "10", "11"))
        self.assertEqual(case_selection("11", "target"), ("11",))
        for invalid in ("", "04-02", "0", "13", "04,,05", "4;5"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                case_selection(invalid, "target")

    def test_outliers_and_correct_tle_are_retained(self):
        row = summarize(timing_samples(), ("10",), 1)[0]
        self.assertEqual(row["samples_per_method"], 2)
        self.assertEqual(row["baseline_max_s"], 3)
        self.assertAlmostEqual(row["baseline_mean_s"], 1.65)
        self.assertAlmostEqual(row["candidate_mean_s"], .105)
        self.assertAlmostEqual(row["delta_candidate_minus_baseline_s"], .105-1.65)
        self.assertEqual(row["candidate_under_100ms_all_samples"], 0)
        self.assertEqual(row["timing_status"], "unstable_samples")

    def test_wrong_output_missing_repeat_and_bad_timing_rejected(self):
        for change in ("WA", "RE", "WALL_TIMEOUT"):
            samples = timing_samples()
            samples[0]["verdict"] = change
            with self.subTest(verdict=change), self.assertRaises(ValueError):
                summarize(samples, ("10",), 1)
        with self.assertRaises(ValueError):
            summarize(timing_samples()[:-1], ("10",), 1)
        for value in ("0", "nan", "inf", "-1"):
            samples = timing_samples()
            samples[0]["compute_s"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(samples, ("10",), 1)

    def test_profile_samples_never_enter_timing_ratios(self):
        samples = timing_samples()
        profiled = copy.deepcopy(samples)
        for sample in profiled:
            sample.update(profiled=1, compute_s="999", profile_status="ok", profile_kernel="new_kernel")
            sample.update({key: "123" for key in PROFILE_TIMES})
        row = summarize(samples+profiled, ("10",), 1)[0]
        self.assertAlmostEqual(row["baseline_mean_s"], 1.65)
        phases = phase_summary(samples+profiled, ("10",), 1)
        self.assertEqual(len(phases), 2)
        self.assertEqual(phases[0]["observations"], 2)
        self.assertEqual(phases[0]["kernels_ms_mean"], 123)
        with self.assertRaises(ValueError):
            phase_summary(samples+profiled[:-1], ("10",), 1)

    def test_phase_parser_accepts_new_kernels_but_checks_values(self):
        record = "[PP_PROFILE] kernel=rolling " + " ".join(key+"=1.25" for key in PROFILE_TIMES)
        self.assertEqual(profile_fields(record)["profile_kernel"], "rolling")
        for invalid in (record+"\n"+record, "", record+" kernels_ms=1.25",
                        record.replace("kernels_ms=1.25", "kernels_ms=nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                profile_fields(invalid)

    def test_cpu_gate_cannot_silently_use_missing_or_partial_suite(self):
        self.assertEqual(validate_gate("PASS 17/17 randomized cases\n"), 17)
        with self.assertRaises(ValueError):
            validate_gate("PASS 17/17 randomized cases\n", 18)
        for text in ("PASS 0/0", "PASS 16/17", "no suite", "PASS 1/1\nPASS 2/2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_gate(text)

    def test_global_dispatch_requested_flags_and_selected_kernel_agree(self):
        with patch.dict(os.environ):
            env = configure("combined", True, True)
            for kernel, chosen in (("dense_multistate", "multistate"), ("rolling", "rolling"),
                                   ("sparse_rolling", "rolling"), ("sparse_bits", "sparse_bits"),
                                   ("dense_bits", None)):
                text = "[PP_GLOBAL] " + " ".join(
                    f"{feature}_requested=1 {feature}_selected={int(feature == chosen)}"
                    for feature in ("multistate", "rolling", "sparse_bits"))
                fields = verify_global(text, env, kernel)
                self.assertEqual(fields["global_multistate_requested"], "1")
                for invalid in ("", text+"\n"+text, text+" multistate_requested=1",
                                text.replace("rolling_requested=1", "rolling_requested=0")):
                    with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                        verify_global(invalid, env, kernel)
            disabled = configure("combined", False, True)
            with self.assertRaises(ValueError):
                verify_global(text, disabled, "sparse_bits")

    def test_cpu_gate_requires_dispatch_coverage_and_feature_selection(self):
        with patch.dict(os.environ):
            env = configure("multistate", True, True)
            profile = "[PP_PROFILE] kernel=naive "+" ".join(key+"=1" for key in PROFILE_TIMES)
            record = "[PP_GLOBAL] multistate_requested=1 multistate_selected=0 rolling_requested=0 rolling_selected=0 sparse_bits_requested=0 sparse_bits_selected=0"
            with self.assertRaises(ValueError):
                verify_gate_dispatch(profile+"\n"+record, env, 1, "multistate")
            with self.assertRaises(ValueError):
                verify_gate_dispatch(profile+"\n"+record, env, 2)
            profile = profile.replace("kernel=naive", "kernel=dense_multistate")
            record = record.replace("multistate_selected=0", "multistate_selected=1")
            self.assertEqual(verify_gate_dispatch(profile+"\n"+record, env, 1, "multistate")["multistate_selected_cases"], 1)

    def test_binary_and_source_snapshot_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/"BUILT").write_text("done")
            binary, source = folder/"binary", folder/"source.cu"
            binary.write_bytes(b"compiled")
            source.write_text("source")
            manifest = dict(binaries=dict(candidate=dict(path="binary", sha256=sha256(binary))),
                            source_hashes={"source.cu": sha256(source)})
            (folder/"manifest.json").write_text(json.dumps(manifest))
            self.assertIn("candidate", verify_manifest(folder)[1])
            binary.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_manifest(folder)
            binary.write_bytes(b"compiled")
            source.write_text("changed")
            with self.assertRaises(ValueError):
                verify_manifest(folder)

    def run_fake_comparison(self, folder, wrong_output=False):
        build = folder/"build"
        build.mkdir()
        (build/"BUILT").write_text("done")
        for name in ("revision.log", "status.log", "working-diff.log", "compiler-version.log"):
            (build/name).write_text("fixture\n")
        binaries = {}
        for name in ("baseline", "candidate", "gate"):
            path = build/name
            path.write_text("fake binary " + name)
            binaries[name] = dict(path=name, sha256=sha256(path))
        manifest = dict(baseline_revision="old", candidate_revision="new", binaries=binaries, source_hashes={})
        (build/"manifest.json").write_text(json.dumps(manifest))
        (folder/"cases").mkdir()
        for case in ("04", "11"):
            (folder/"cases"/(case+".txt")).write_text("64 64\nR5,C2,M0,S2..3,B3..3,NM\ndead\n5\n0\n")
            (folder/"cases"/(case+".golden.txt")).write_text("golden")
        def fake_capture(argv, log, **unused):
            count = GATE_COUNTS.get(argv[1], 0)
            if not argv[0].endswith("gate"):
                log.write_text("fixture\n")
                return
            selected = {"--multistate": "multistate", "--rolling": "rolling", "--sparse-bits": "sparse_bits"}.get(argv[1])
            kernel = {"multistate": "dense_multistate", "rolling": "rolling", "sparse_bits": "sparse_bits"}.get(selected, "naive")
            profile = "[PP_PROFILE] kernel="+kernel+" "+" ".join(key+"=1" for key in PROFILE_TIMES)+"\n"
            dispatch = "[PP_GLOBAL] "+" ".join(
                f"{feature}_requested={os.environ['PP_'+feature.upper()]} {feature}_selected={int(feature == selected)}"
                for feature in ("multistate", "rolling", "sparse_bits"))+"\n"
            log.write_text((profile+dispatch)*count+f"PASS {count}/{count} fixture\n")
        def fake_git(destination):
            for name in ("revision.log", "status.log", "working-diff.log"):
                (destination/name).write_text("fixture\n")
        def fake_case(args):
            path = Path(args.inputs[0])
            output = Path(args.outdir)
            output.mkdir()
            profiled = os.environ["PP_PROFILE"] == "1"
            candidate = Path(args.binary).name == "candidate"
            elapsed = .01 if path.stem == "11" else .04 if candidate else .3
            verdict = "WA" if wrong_output else "TLE" if elapsed > .2 else "AC"
            kernel = "dense_bits" if path.stem == "11" else "rolling"
            row = dict(case=str(path), verdict=verdict, compute_s=".99" if profiled else elapsed,
                       wall_s=1, limit_s=.2, profile_status="invalid" if profiled else "absent",
                       profile_kernel="", **{key: "" for key in PROFILE_TIMES})
            with (output/"results.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row), delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            raw = "[PP_PROFILE] kernel="+kernel+" "+" ".join(key+"=1" for key in PROFILE_TIMES)+"\n" if profiled else ""
            if profiled and candidate:
                raw += "[PP_GLOBAL] "+" ".join(
                    f"{feature}_requested=1 {feature}_selected={int(kernel == 'rolling' and feature == 'rolling')}"
                    for feature in ("multistate", "rolling", "sparse_bits"))+"\n"
            (output/f"00-{path.stem}.log").write_text(raw)
            (output/f"00-{path.stem}.check.log").write_text(verdict)
            return int(verdict != "AC")
        args = argparse.Namespace(build_dir=build, outdir=folder/"results", method="combined", cases="04",
                                  group="target", blocks=1, profile_blocks=1, budget_seconds=420,
                                  wall_timeout=20, checker="fake-checker")
        previous = Path.cwd()
        try:
            os.chdir(folder)
            with patch.dict(os.environ, SLURM_JOB_ID="fixture"), \
                    patch("global_compare.shutil.which", return_value="fake-checker"), \
                    patch("global_compare.capture", side_effect=fake_capture), \
                    patch("global_compare.record_git", side_effect=fake_git), \
                    patch("global_compare.run_cases", side_effect=fake_case), \
                    contextlib.redirect_stdout(io.StringIO()):
                if wrong_output:
                    with self.assertRaises(ValueError):
                        compare(args)
                else:
                    compare(args)
        finally:
            os.chdir(previous)
        return folder/"results"

    def test_complete_pipeline_preserves_correct_tle_and_separate_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            results = self.run_fake_comparison(Path(temp))
            self.assertTrue((results/"COMPLETE").is_file())
            with (results/"samples.tsv").open(newline="") as stream:
                samples = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(samples), 16)
            self.assertEqual(sum(row["verdict"] == "TLE" for row in samples), 4)
            with (results/"summary.tsv").open(newline="") as stream:
                summary = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["case"] for row in summary], ["04", "11"])
            self.assertAlmostEqual(float(summary[0]["baseline_mean_s"]), .3)
            self.assertAlmostEqual(float(summary[0]["candidate_mean_s"]), .04)
            environments = json.loads((results/"effective-environments.json").read_text())
            sparse_gates = [row for row in environments if row.get("option") == "--sparse-bits"]
            self.assertEqual(len(sparse_gates), 2)
            self.assertTrue(all(row["env"]["PP_ACTIVE"] == "1" for row in sparse_gates))
            self.assertEqual({row["env"]["PP_SUPPORT"] for row in sparse_gates}, {"bbox", "tile"})

    def test_failed_pipeline_keeps_wrong_output_record_and_no_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            results = self.run_fake_comparison(Path(temp), wrong_output=True)
            self.assertFalse((results/"COMPLETE").exists())
            with (results/"samples.tsv").open(newline="") as stream:
                samples = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["verdict"], "WA")


if __name__ == "__main__":
    unittest.main()
