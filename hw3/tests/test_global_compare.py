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
from global_compare import (ORIGINAL_BASELINE_REVISION, comparison_binaries, gate_plan, require_original_baseline,
                            same_fixed_source, verify_gate_dispatch, verify_sparse_next)
from run_cases import PROFILE_TIMES, parse_profile


def timing_samples():
    return [dict(case_id="10", method=method, profiled=0, verdict=verdict, compute_s=value)
            for method, verdict, value in (("baseline", "AC", ".3"), ("candidate", "AC", ".1"),
                                           ("candidate", "AC", ".11"), ("baseline", "TLE", "3"))]


class GlobalComparisonTests(unittest.TestCase):
    def test_regression_control_requires_exact_original_baseline_revision(self):
        manifest = dict(baseline_revision=ORIGINAL_BASELINE_REVISION, candidate_revision="current")
        before = copy.deepcopy(manifest)
        require_original_baseline(manifest)
        self.assertEqual(manifest, before)
        for revision in ("3b88c9dccc89cb0060fbcb90991fa07a19ed7a33", "1352bf8", None, "", "current"):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "requires frozen baseline"):
                require_original_baseline(dict(baseline_revision=revision))
        with self.assertRaisesRegex(ValueError, "missing"):
            require_original_baseline({})

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

    def test_disabled_method_clears_all_new_features_and_keeps_only_quick_gates(self):
        features = ("PP_MULTISTATE", "PP_ROLLING", "PP_SPARSE_BITS", "PP_SPARSE_WARP",
                    "PP_SPARSE_WIDE", "PP_SPARSE_OUTPUT")
        with patch.dict(os.environ, {name: "1" for name in features}):
            env = configure("disabled", True)
            self.assertTrue(all(env[name] == "0" for name in features))
            self.assertEqual([item["option"] for item in gate_plan("disabled")],
                             ["--quick", "--quick"])

    def test_baseline_binary_alias_preserves_frozen_manifest_and_records_real_source(self):
        manifest = dict(baseline_revision="old", candidate_revision="new", binaries={
            name: dict(sha256=name+"-hash") for name in ("baseline", "candidate", "gate")})
        binaries = {name: Path(name) for name in manifest["binaries"]}
        original_manifest, original_binaries = copy.deepcopy(manifest), dict(binaries)
        selected, evidence = comparison_binaries(manifest, binaries, "candidate")
        self.assertEqual(selected["baseline"], selected["candidate"])
        self.assertEqual(evidence["baseline"], evidence["candidate"])
        self.assertEqual(evidence["baseline"]["source_revision"], "new")
        self.assertEqual(evidence["baseline"]["source_snapshot"], "candidate-source/hw3")
        self.assertEqual(evidence["baseline"]["binary_sha256"], "candidate-hash")
        self.assertEqual(manifest, original_manifest)
        self.assertEqual(binaries, original_binaries)
        normal, old_evidence = comparison_binaries(manifest, binaries)
        self.assertEqual(normal, binaries)
        self.assertEqual(old_evidence["baseline"]["source_revision"], "old")
        with self.assertRaises(ValueError):
            comparison_binaries(manifest, binaries, "gate")

    def test_next_methods_compare_against_combined_without_leaking_new_flags(self):
        with patch.dict(os.environ, PP_SPARSE_WARP="1", PP_SPARSE_WIDE="1", PP_SPARSE_OUTPUT="1"):
            baseline = configure("sparse-next", False, True, baseline_method="combined")
            candidate = configure("sparse-next", True)
            for key in ("PP_MULTISTATE", "PP_ROLLING", "PP_SPARSE_BITS"):
                self.assertEqual(baseline[key], "1")
                self.assertEqual(candidate[key], "1")
            for key in ("PP_SPARSE_WARP", "PP_SPARSE_WIDE", "PP_SPARSE_OUTPUT"):
                self.assertEqual(baseline[key], "0")
                self.assertEqual(candidate[key], "1")
            for method, chosen in (("sparse-warp", "PP_SPARSE_WARP"),
                                   ("sparse-wide", "PP_SPARSE_WIDE"),
                                   ("sparse-output", "PP_SPARSE_OUTPUT")):
                env = configure(method, True)
                self.assertEqual([key for key in ("PP_SPARSE_WARP", "PP_SPARSE_WIDE", "PP_SPARSE_OUTPUT")
                                  if env[key] == "1"], [chosen])

    def test_next_gate_plan_validates_baseline_and_required_fallbacks(self):
        plan = gate_plan("sparse-next", "combined")
        self.assertEqual([(item["option"], item["role"]) for item in plan[:3]],
                         [("--quick", "disabled"), ("--quick", "baseline"), ("--quick", "candidate")])
        self.assertNotIn("--multistate", [item["option"] for item in plan])
        self.assertEqual({item["required"] for item in plan if item["required"]},
                         {"sparse_warp", "sparse_wide", "sparse_output"})
        wide = [item for item in plan if item["option"] == "--sparse-wide"]
        self.assertEqual(len(wide), 2)
        self.assertEqual({item["overrides"].get("PP_SUPPORT", "bbox") for item in wide}, {"bbox", "tile"})
        pageable = [item for item in plan if item["option"] == "--sparse-pageable"]
        self.assertEqual(pageable[0]["zero"], "sparse_output")
        self.assertEqual(GATE_COUNTS["--sparse-wide"], 200)
        self.assertEqual(GATE_COUNTS["--sparse-warp"], 32)

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
        for kernel in ("sparse_warp", "sparse_wide"):
            self.assertEqual(parse_profile(record.replace("kernel=rolling", "kernel="+kernel))[0], "ok")

    def test_sparse_next_dispatch_requires_requested_kernel_and_output_evidence(self):
        with patch.dict(os.environ):
            env = configure("sparse-next", True, True)
            valid = ("[PP_SPARSE_NEXT] warp_requested=1 warp_selected=0 wide_requested=1 wide_selected=1 "
                     "output_requested=1 output_selected=1 output_reason=enabled clear_bytes=1024 "
                     "scatter_bytes=2048 tile_w=256 tile_h=16")
            self.assertEqual(verify_sparse_next(valid, env, "sparse_wide", 4096)["sparse_next_output_selected"], "1")
            for invalid in ("", valid+"\n"+valid, valid+" tile_h=16",
                            valid.replace("wide_requested=1", "wide_requested=0"),
                            valid.replace("warp_selected=0", "warp_selected=1"),
                            valid.replace("output_reason=enabled", "output_reason=not_mapped"),
                            valid.replace("scatter_bytes=2048", "scatter_bytes=4097"),
                            valid.replace("clear_bytes=1024", "clear_bytes=-1"),
                            valid.replace("tile_h=16", "tile_h=32")):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    verify_sparse_next(invalid, env, "sparse_wide", 4096)
            with self.assertRaises(ValueError):
                verify_sparse_next(valid, env, "sparse_wide")
            fallback = valid.replace("output_selected=1", "output_selected=0").replace(
                "output_reason=enabled", "output_reason=dense_fallback").replace(
                "clear_bytes=1024", "clear_bytes=0").replace("scatter_bytes=2048", "scatter_bytes=0")
            self.assertEqual(verify_sparse_next(fallback, env, "sparse_wide", 4096)["sparse_next_output_selected"], "0")
            with self.assertRaises(ValueError):
                verify_sparse_next(fallback.replace("clear_bytes=0", "clear_bytes=1"), env, "sparse_wide", 4096)
            with self.assertRaises(ValueError):
                verify_sparse_next(fallback.replace("output_reason=dense_fallback", "output_reason=disabled"), env, "sparse_wide", 4096)
            disabled = configure("combined", True, True)
            self.assertEqual(verify_sparse_next("", disabled, "rolling")["sparse_next_warp_selected"], "0")

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
                                   ("sparse_warp", None), ("sparse_wide", None),
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

    def test_next_gate_rejects_missing_records_unselected_feature_and_pageable_selection(self):
        with patch.dict(os.environ):
            env = configure("sparse-next", True, True)
            profile = "[PP_PROFILE] kernel=sparse_wide cells=4096 "+" ".join(key+"=1" for key in PROFILE_TIMES)
            old = "[PP_GLOBAL] multistate_requested=1 multistate_selected=0 rolling_requested=1 rolling_selected=0 sparse_bits_requested=1 sparse_bits_selected=0"
            new = ("[PP_SPARSE_NEXT] warp_requested=1 warp_selected=0 wide_requested=1 wide_selected=1 "
                   "output_requested=1 output_selected=1 output_reason=enabled clear_bytes=0 scatter_bytes=0 tile_w=256 tile_h=16")
            text = "\n".join((profile, old, new))
            self.assertEqual(verify_gate_dispatch(text, env, 1, "sparse_output")["sparse_wide_selected_cases"], 1)
            with self.assertRaises(ValueError):
                verify_gate_dispatch("\n".join((profile, old)), env, 1)
            with self.assertRaises(ValueError):
                verify_gate_dispatch(text+"\n"+new, env, 1)
            with self.assertRaises(ValueError):
                verify_gate_dispatch(text, env, 1, zero_feature="sparse_output")
            fallback = text.replace("output_selected=1", "output_selected=0").replace("output_reason=enabled", "output_reason=not_pinned_host")
            self.assertEqual(verify_gate_dispatch(fallback, env, 1, zero_feature="sparse_output")["sparse_output_selected_cases"], 0)
            with self.assertRaises(ValueError):
                verify_gate_dispatch(fallback, env, 1, "sparse_output")

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

    def run_fake_comparison(self, folder, wrong_output=False, method="combined", baseline_method="disabled",
                            baseline_binary="baseline"):
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
        def records(kernel, output_selected=False, output_reason=None, include_next=True):
            result = "[PP_PROFILE] kernel="+kernel+" cells=4096 "+" ".join(key+"=1" for key in PROFILE_TIMES)+"\n"
            result += "[PP_GLOBAL] "+" ".join(
                f"{feature}_requested={os.environ['PP_'+feature.upper()]} "
                f"{feature}_selected={int(kernel == {'multistate': 'dense_multistate', 'rolling': 'rolling', 'sparse_bits': 'sparse_bits'}[feature])}"
                for feature in ("multistate", "rolling", "sparse_bits"))+"\n"
            if include_next:
                result += "[PP_SPARSE_NEXT] "+" ".join(
                    f"{feature}_requested={os.environ['PP_SPARSE_'+feature.upper()]} "
                    f"{feature}_selected={int(output_selected if feature == 'output' else kernel == 'sparse_'+feature)}"
                    for feature in ("warp", "wide", "output"))
                reason = output_reason or ("enabled" if output_selected else
                                           "not_sparse" if os.environ["PP_SPARSE_OUTPUT"] == "1" else "disabled")
                result += f" output_reason={reason} clear_bytes={64 if output_selected else 0} "
                result += f"scatter_bytes={128 if output_selected else 0} "
                result += "tile_w=256 tile_h=16\n" if kernel == "sparse_wide" else "tile_w=64 tile_h=32\n"
            return result
        def fake_capture(argv, log, **unused):
            count = GATE_COUNTS.get(argv[1], 0)
            if not argv[0].endswith("gate"):
                log.write_text("fixture\n")
                return
            selected = {"--multistate": "multistate", "--rolling": "rolling", "--sparse-bits": "sparse_bits"}.get(argv[1])
            kernel = {"multistate": "dense_multistate", "rolling": "rolling", "sparse_bits": "sparse_bits"}.get(selected, "naive")
            if argv[1] == "--sparse-wide":
                kernel = "sparse_wide"
            elif argv[1] == "--sparse-warp" or (argv[1] == "--rolling" and os.environ["PP_SPARSE_WARP"] == "1"):
                kernel = "sparse_warp"
            output_selected = argv[1] == "--sparse" and os.environ["PP_SPARSE_OUTPUT"] == "1"
            if output_selected or argv[1] == "--sparse-pageable":
                kernel = "sparse_bits"
            reason = "not_pinned_host" if argv[1] == "--sparse-pageable" else None
            log.write_text(records(kernel, output_selected, reason)*count+f"PASS {count}/{count} fixture\n")
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
            kernel = "dense_bits" if path.stem == "11" else (
                "sparse_warp" if candidate and os.environ["PP_SPARSE_WARP"] == "1" else
                "rolling" if os.environ["PP_ROLLING"] == "1" else
                "dense_multistate" if os.environ["PP_MULTISTATE"] == "1" else "naive")
            row = dict(case=str(path), verdict=verdict, compute_s=".99" if profiled else elapsed,
                       wall_s=1, limit_s=.2, profile_status="invalid" if profiled else "absent",
                       profile_kernel="", **{key: "" for key in PROFILE_TIMES})
            with (output/"results.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row), delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            raw = records(kernel, include_next=candidate) if profiled else ""
            (output/f"00-{path.stem}.log").write_text(raw)
            (output/f"00-{path.stem}.check.log").write_text(verdict)
            return int(verdict != "AC")
        args = argparse.Namespace(build_dir=build, outdir=folder/"results", method=method, baseline_method=baseline_method, cases="04",
                                  baseline_binary=baseline_binary,
                                  group="target", blocks=1, profile_blocks=1, budget_seconds=420,
                                  wall_timeout=20, checker="fake-checker")
        previous = Path.cwd()
        console = io.StringIO()
        try:
            os.chdir(folder)
            with patch.dict(os.environ, SLURM_JOB_ID="fixture"), \
                    patch("global_compare.shutil.which", return_value="fake-checker"), \
                    patch("global_compare.capture", side_effect=fake_capture), \
                    patch("global_compare.record_git", side_effect=fake_git), \
                    patch("global_compare.run_cases", side_effect=fake_case), \
                    contextlib.redirect_stdout(console):
                if wrong_output:
                    with self.assertRaises(ValueError):
                        compare(args)
                else:
                    compare(args)
        finally:
            os.chdir(previous)
        (folder/"captured-console.log").write_text(console.getvalue())
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
            self.assertNotIn("[PP_SPARSE_NEXT]", (Path(temp)/"captured-console.log").read_text())

    def test_failed_pipeline_keeps_wrong_output_record_and_no_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            results = self.run_fake_comparison(Path(temp), wrong_output=True)
            self.assertFalse((results/"COMPLETE").exists())
            with (results/"samples.tsv").open(newline="") as stream:
                samples = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["verdict"], "WA")

    def test_next_pipeline_uses_combined_baseline_and_checks_pageable_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            results = self.run_fake_comparison(Path(temp), method="sparse-next", baseline_method="combined")
            self.assertTrue((results/"COMPLETE").is_file())
            configurations = json.loads((results/"configuration.json").read_text())
            self.assertEqual(configurations["baseline_method"], "combined")
            envs = json.loads((results/"effective-environments.json").read_text())
            for item in envs:
                if item["method"] == "baseline":
                    self.assertEqual(item["env"]["PP_ROLLING"], "1")
                    self.assertEqual(item["env"]["PP_SPARSE_WARP"], "0")
            with (results/"gates.tsv").open(newline="") as stream:
                gates = list(csv.DictReader(stream, delimiter="\t"))
            pageable = next(row for row in gates if row["option"] == "--sparse-pageable")
            self.assertEqual(pageable["sparse_output_selected_cases"], "0")
            for feature in ("sparse_warp", "sparse_wide", "sparse_output"):
                self.assertTrue(any(int(row[feature+"_selected_cases"]) > 0 for row in gates))
            console = (Path(temp)/"captured-console.log").read_text()
            self.assertEqual(console.count("[PP_SPARSE_NEXT]"), 4)
            self.assertIn("Case 04: [PP_SPARSE_NEXT]", console)

    def test_disabled_pipeline_runs_current_binary_without_new_features(self):
        with tempfile.TemporaryDirectory() as temp:
            results = self.run_fake_comparison(Path(temp), method="disabled")
            self.assertTrue((results/"COMPLETE").is_file())
            envs = json.loads((results/"effective-environments.json").read_text())
            for item in envs:
                self.assertTrue(all(item["env"][name] == "0" for name in
                                    ("PP_MULTISTATE", "PP_ROLLING", "PP_SPARSE_BITS", "PP_SPARSE_WARP",
                                     "PP_SPARSE_WIDE", "PP_SPARSE_OUTPUT")))
            configuration = json.loads((results/"configuration.json").read_text())
            self.assertEqual(configuration["baseline_revision"], "old")
            self.assertEqual(configuration["binary_provenance"]["candidate"]["source_revision"], "new")

    def test_same_binary_pipeline_attributes_flags_without_claiming_old_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            results = self.run_fake_comparison(Path(temp), method="multistate", baseline_binary="candidate")
            self.assertTrue((results/"COMPLETE").is_file())
            configuration = json.loads((results/"configuration.json").read_text())
            self.assertEqual(configuration["baseline_revision"], "new")
            self.assertEqual(configuration["frozen_baseline_revision"], "old")
            evidence = configuration["binary_provenance"]
            self.assertEqual(evidence["baseline"], evidence["candidate"])
            envs = json.loads((results/"effective-environments.json").read_text())
            rounds = [item for item in envs if item["kind"] == "round"]
            for item in rounds:
                self.assertEqual(item["binary_provenance"], evidence["candidate"])
                self.assertEqual(item["env"]["PP_MULTISTATE"], str(int(item["method"] == "candidate")))
            with (results/"samples.tsv").open(newline="") as stream:
                samples = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual({row["frozen_binary"] for row in samples}, {"candidate"})
            self.assertEqual({row["source_revision"] for row in samples}, {"new"})
            self.assertEqual(len({row["binary_sha256"] for row in samples}), 1)
            console = (Path(temp)/"captured-console.log").read_text()
            self.assertIn("Baseline: new (frozen candidate binary)", console)
            self.assertNotIn("Baseline: old", console)


if __name__ == "__main__":
    unittest.main()
