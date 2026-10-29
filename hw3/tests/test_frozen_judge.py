import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from frozen_judge import judge_environment, run_judge, verified_judge_source
from global_compare import ORIGINAL_BASELINE_REVISION, configure, sha256


def frozen_build(root):
    source = root/"candidate-source"/"hw3"
    source.mkdir(parents=True)
    hashes = {}
    for name in ("hw3-1.cu", "main.cu", "ca.hpp", "r1_multistate.hpp", "r1_multistate.cuh"):
        path = source/name
        path.write_text("frozen " + name, encoding="utf-8")
        hashes[path.relative_to(root).as_posix()] = sha256(path)
    binaries = {}
    for name in ("baseline", "candidate", "gate"):
        path = root/(name + "-binary")
        path.write_text(name, encoding="utf-8")
        binaries[name] = dict(path=path.name, sha256=sha256(path))
    manifest = dict(baseline_revision=ORIGINAL_BASELINE_REVISION,
                    candidate_base_revision=ORIGINAL_BASELINE_REVISION,
                    candidate_revision="generating-head", candidate_variant="original-multistate",
                    source_hashes=hashes, binaries=binaries)
    (root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root/"BUILT").touch()
    return source, manifest


class FrozenJudgeTests(unittest.TestCase):
    def test_environment_clears_unknown_overrides_and_matches_measured_method(self):
        inherited = dict(PATH="keep", PP_PROFILE="1", PP_KERNEL="naive", PP_FUTURE="1")
        environment = judge_environment(inherited)
        self.assertEqual(inherited["PP_PROFILE"], "1")
        self.assertEqual(environment["PATH"], "keep")
        self.assertNotIn("PP_FUTURE", environment)
        self.assertNotIn("PP_KERNEL", environment)
        with patch.dict(os.environ, inherited, clear=True):
            measured = configure("multistate", True, profiled=False)
        self.assertEqual({key: value for key, value in environment.items() if key.startswith("PP_")}, measured)

    def test_source_identity_rejects_working_variant_wrong_base_and_mutated_header(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, manifest = frozen_build(root)
            self.assertEqual(verified_judge_source(root)[1], source.resolve())
            for changed in (dict(manifest, candidate_variant="working"),
                            dict(manifest, candidate_base_revision="other"),
                            dict(manifest, baseline_revision="1352bf8")):
                (root/"manifest.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    verified_judge_source(root)
            (root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (source/"r1_multistate.cuh").write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source snapshot changed"):
                verified_judge_source(root)

    def test_existing_slurm_allocation_is_rejected_before_launch(self):
        with patch.dict(os.environ, SLURM_JOB_ID="already-allocated"), patch("frozen_judge.subprocess.Popen") as launch:
            with self.assertRaisesRegex(ValueError, "creates its own job"):
                run_judge("missing-build")
            launch.assert_not_called()

    def test_judge_uses_verified_source_cwd_records_exact_flags_and_leaves_root_untouched(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"PATH": "cluster", "PP_PROFILE": "1"}, clear=True):
            root = Path(temp)
            build = root/"build"
            build.mkdir()
            source, manifest = frozen_build(build)
            working = root/"hw3-1.cu"
            working.write_text("working production", encoding="utf-8")
            class FakeJudge:
                stdout = io.StringIO("nvcc main.cu hw3-1.cu -o hw3-1\njudging 12 case(s)\n")
                def wait(self):
                    return 0
            with patch("frozen_judge.shutil.which", return_value="/judge/hw3-1-judge"), \
                    patch("frozen_judge.subprocess.Popen", return_value=FakeJudge()) as launch, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = run_judge(build, log_root=root/"logs")
            self.assertEqual(result, 0)
            self.assertEqual(launch.call_args.args, (["/judge/hw3-1-judge"],))
            self.assertEqual(launch.call_args.kwargs["cwd"], source.resolve())
            self.assertEqual(launch.call_args.kwargs["env"]["PP_MULTISTATE"], "1")
            self.assertEqual(launch.call_args.kwargs["env"]["PP_PROFILE"], "0")
            self.assertEqual(working.read_text(encoding="utf-8"), "working production")
            logdir, = (root/"logs").iterdir()
            record = json.loads((logdir/"configuration.json").read_text(encoding="utf-8"))
            self.assertEqual(record["candidate_variant"], "original-multistate")
            self.assertEqual(record["candidate_source_sha256"], manifest["source_hashes"]["candidate-source/hw3/hw3-1.cu"])
            self.assertEqual(record["source_and_frozen_binary_hashes_after_judge"], "unchanged")
            self.assertTrue((logdir/"JUDGE_EXITED").is_file())
            self.assertFalse((logdir/"COMPLETE").exists())
            self.assertIn("judging 12", (logdir/"judge-output.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
