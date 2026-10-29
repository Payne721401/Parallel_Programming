"""Run the official judge from verified original-plus-multistate frozen sources.

The user invokes this on the login node: hw3-1-judge builds those sources and
requests its own Slurm job. This helper never allocates a GPU or selects a
prebuilt binary. It cannot establish which environment the remote judge exports;
the complete intended environment and judge output are retained for inspection.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from global_compare import require_original_baseline, safe_environment, sha256, verify_manifest


def judge_environment(inherited):
    """Match the unprofiled multistate-only comparison, clearing unknown PP_* too."""
    environment = {key: value for key, value in inherited.items() if not key.startswith("PP_")}
    environment.update(PP_ARCH="baseline", PP_GRAPH="0", PP_EVOLVE="0",
                       PP_PROFILE="0", PP_SUPPORT="bbox", PP_BITPLANE="auto",
                       PP_BIT_LOGIC="compact", PP_BIT_ROWS="4", PP_MULTISTATE="1",
                       PP_ROLLING="0", PP_SPARSE_BITS="0", PP_SPARSE_WARP="0",
                       PP_SPARSE_WIDE="0", PP_SPARSE_OUTPUT="0", PP_ARENA="0")
    return environment


def verified_judge_source(build_dir):
    folder = Path(build_dir).resolve()
    manifest, binaries = verify_manifest(folder)
    require_original_baseline(manifest)
    if manifest.get("candidate_variant") != "original-multistate":
        raise ValueError("requires candidate variant original-multistate; this build contains "
                         + str(manifest.get("candidate_variant", "working")))
    if manifest.get("candidate_base_revision") != manifest["baseline_revision"]:
        raise ValueError("candidate base does not match the verified original baseline")
    source = (folder/"candidate-source"/"hw3").resolve()
    try:
        source.relative_to(folder)
    except ValueError:
        raise ValueError("frozen candidate source resolves outside the build directory")
    for name in ("hw3-1.cu", "main.cu", "ca.hpp", "r1_multistate.hpp", "r1_multistate.cuh"):
        relative = "candidate-source/hw3/" + name
        if relative not in manifest["source_hashes"] or not (source/name).is_file():
            raise ValueError("missing verified candidate source: " + name)
    return folder, source, manifest, binaries


def run_judge(build_dir, log_root=None):
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("run from the login node without srun/sbatch; hw3-1-judge creates its own job")
    folder, source, manifest, binaries = verified_judge_source(build_dir)
    judge = shutil.which("hw3-1-judge")
    if judge is None:
        raise ValueError("hw3-1-judge is not on PATH; use the cluster login shell with judge setup loaded")
    environment = judge_environment(os.environ)
    pp_environment = {key: environment[key] for key in sorted(environment) if key.startswith("PP_")}
    root = Path(log_root) if log_root is not None else Path(__file__).resolve().parents[1]/"tmp"
    logdir = root/("frozen-judge-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()))
    logdir.mkdir(parents=True, exist_ok=False)
    provenance = dict(build_dir=str(folder), source_directory=str(source), judge_command=[judge],
                      candidate_variant=manifest["candidate_variant"],
                      original_base_revision=manifest["candidate_base_revision"],
                      generating_head=manifest["candidate_revision"],
                      frozen_manifest_sha256=sha256(folder/"manifest.json"),
                      candidate_source_sha256=manifest["source_hashes"]["candidate-source/hw3/hw3-1.cu"],
                      comparison_binary_sha256=manifest["binaries"]["candidate"]["sha256"],
                      comparison_binary_path=str(binaries["candidate"]),
                      comparison_binary_is_not_passed_to_judge=True,
                      pp_environment=pp_environment, launch_environment=safe_environment())
    (logdir/"configuration.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Frozen judge logs: " + str(logdir), flush=True)
    print("Verified source variant: " + manifest["candidate_variant"], flush=True)
    print("Original base: " + manifest["candidate_base_revision"], flush=True)
    print("Judge source directory: " + str(source), flush=True)
    print("Source SHA256: " + provenance["candidate_source_sha256"], flush=True)
    print("Effective launch flags: " + " ".join(key+"="+value for key, value in pp_environment.items()), flush=True)
    print("The official judge rebuilds these sources and allocates its own Slurm job.", flush=True)
    print("It runs all public cases and updates the scoreboard; the documented cooldown applies.", flush=True)
    with (logdir/"judge-output.log").open("w", encoding="utf-8") as output:
        process = subprocess.Popen([judge], cwd=source, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
            output.flush()
        returncode = process.wait()
    provenance["judge_exit_code"] = returncode
    try:
        verify_manifest(folder)
        provenance["source_and_frozen_binary_hashes_after_judge"] = "unchanged"
    except (OSError, ValueError, KeyError) as error:
        provenance["source_and_frozen_binary_hashes_after_judge"] = str(error)
        print("Post-judge frozen hash check failed: " + str(error), file=sys.stderr, flush=True)
        if returncode == 0:
            returncode = 1
    (logdir/"configuration.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    # A zero command exit is not proof of 12/12 AC: retain the actual judge table.
    (logdir/"JUDGE_EXITED").write_text("Judge command exit: " + str(provenance["judge_exit_code"]) + "\n")
    print("Judge output retained: " + str(logdir/"judge-output.log"), flush=True)
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True,
                        help="existing original-multistate frozen build, e.g. tmp/lean-multistate-build-v1")
    args = parser.parse_args()
    try:
        return run_judge(args.build_dir)
    except (OSError, ValueError, KeyError) as error:
        print("Frozen judge handoff failed: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
