"""Check fixed FourRoom inference with golden context at every prediction step.

This measures teacher-forced inference, not a free-running rollout or planning
success. The provisional absolute epsilon needs calibration on the grading GPU.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch


ROOT = Path(__file__).resolve().parent
LEVELS = (1, 2, 3)
LATENT_DIM = 192


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive_epsilon(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("epsilon must be finite and positive")
    return number


def _tensor(value, shape, description):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError(f"{description}: expected an FP32 tensor")
    if tuple(value.shape) != tuple(shape) or value.device.type != "cpu":
        raise ValueError(f"{description}: expected CPU shape {tuple(shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{description}: non-finite fixture values")


def load_fixtures(folder: Path, manifest: dict, case_id: str | None = None):
    cases_path, golden_path = folder / "cases.pt", folder / "golden.pt"
    artifact_path = folder / "manifest.json"
    if not all(path.is_file() for path in (cases_path, golden_path, artifact_path)):
        raise FileNotFoundError(
            "Missing fixed teacher-forcing fixtures. Restore assets/check/ "
            "from the released student package; golden preparation is TA-only. "
            f"Expected cases.pt, golden.pt and manifest.json under {folder}."
        )
    artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifacts.get("format") != "fourroom_check_artifacts_v1":
        raise ValueError("Unsupported check artifact manifest")
    files = artifacts.get("files", {})
    if set(files) != {"cases.pt", "golden.pt"}:
        raise ValueError("Check artifact manifest must identify cases.pt and golden.pt")
    hashes = {"manifest_sha256": digest(artifact_path)}
    for path, name in ((cases_path, "cases"), (golden_path, "golden")):
        value = digest(path)
        entry = files[path.name]
        if value != entry.get("sha256") or path.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"Check artifact SHA256/size mismatch: {path.name}")
        hashes[f"{name}_sha256"] = value
    for name in ("architecture_sha256", "weights_sha256"):
        if artifacts.get(name) != manifest[name]:
            raise ValueError(f"Check artifact fixed-asset identity mismatch: {name}")
    cases = torch.load(cases_path, map_location="cpu", weights_only=True)
    golden = torch.load(golden_path, map_location="cpu", weights_only=True)
    if not isinstance(cases, dict) or cases.get("format") != "fourroom_check_cases_v1":
        raise ValueError("Unsupported check-case format")
    if not isinstance(golden, dict) or golden.get("format") != "fourroom_teacher_forced_golden_v1":
        raise ValueError("Unsupported teacher-forcing golden format")
    if golden.get("cases_sha256") != hashes["cases_sha256"]:
        raise ValueError("Golden answers were generated for a different cases.pt")
    for name in ("architecture_sha256", "weights_sha256"):
        if cases.get(name) != manifest[name] or golden.get(name) != manifest[name]:
            raise ValueError(f"Fixture fixed-asset identity mismatch: {name}")
    if cases.get("seed") != 42:
        raise ValueError("Expected check fixture seed 42")
    entries = cases.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Empty or invalid check-case list")
    ids = [case.get("id") for case in entries]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid or repeated check-case IDs")
    if not isinstance(golden.get("cases"), dict) or set(golden["cases"]) != set(ids):
        raise ValueError("Golden case IDs do not match the input cases")
    for case in entries:
        label = case["id"]
        pixels = case.get("pixels")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[0] < 1:
            raise ValueError(f"{label}: invalid image batch")
        batch = pixels.shape[0]
        _tensor(pixels, (batch, 3, 224, 224), f"{label}/pixels")
        actions = case.get("actions")
        reference = golden["cases"][label]
        if not isinstance(actions, dict) or set(actions) != set(LEVELS):
            raise ValueError(f"{label}: expected action banks for levels 1, 2, 3")
        if any(not isinstance(reference.get(k), dict) or set(reference[k]) != set(LEVELS)
               for k in ("initial", "rollouts")):
            raise ValueError(f"{label}: invalid golden level dictionaries")
        for level in LEVELS:
            bank = actions[level]
            if (not isinstance(bank, torch.Tensor) or bank.ndim != 4
                    or bank.shape[0] != batch or min(bank.shape[1:3]) < 1):
                raise ValueError(f"{label}/L{level}: invalid action-bank shape")
            samples, horizon = bank.shape[1:3]
            _tensor(bank, (batch, samples, horizon, 10 if level == 1 else 4),
                    f"{label}/L{level}/actions")
            _tensor(reference["initial"][level], (batch, 1, LATENT_DIM),
                    f"{label}/L{level}/initial")
            _tensor(reference["rollouts"][level], (batch, samples, horizon + 1, LATENT_DIM),
                    f"{label}/L{level}/rollouts")
            expanded = reference["initial"][level][:, None].expand(batch, samples, 1, LATENT_DIM)
            if not torch.equal(reference["rollouts"][level][:, :, :1], expanded):
                raise ValueError(f"{label}/L{level}: golden initial state disagrees with rollout")
    selected = [case for case in entries if case_id is None or case["id"] == case_id]
    if not selected:
        raise ValueError(f"Unknown check case: {case_id}")
    return selected, golden["cases"], {**hashes, "seed": cases["seed"],
                                     "golden_environment": golden.get("environment")}


def compare_tensor(value, target, epsilon: float, device: str = "cuda") -> dict:
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or value.device.type != torch.device(device).type
            or tuple(value.shape) != tuple(target.shape)):
        return {"passed": False, "reason": "Wrong output type/device/dtype/shape"}
    if not bool(torch.isfinite(value).all()):
        return {"passed": False, "reason": "Non-finite output"}
    actual, expected = value.detach().cpu().double(), target.detach().cpu().double()
    difference = (actual - expected).abs()
    relative = torch.linalg.vector_norm(difference, dim=-1) / torch.linalg.vector_norm(
        expected, dim=-1
    ).clamp_min(1e-12)
    maximum = float(difference.max())
    return {"passed": maximum <= epsilon, "max_abs": maximum,
            "max_relative_l2": float(relative.max())}


class CudaTimer:
    """One forward only; input copies and validation occur outside this call."""

    def __call__(self, forward):
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        result = forward()
        end.record()
        torch.cuda.synchronize()
        return result, {"wall_ms": (time.perf_counter() - start) * 1000,
                        "cuda_event_ms": begin.elapsed_time(end)}


def check_case(model, case: dict, golden: dict, *, encode, predict, epsilon: float,
               device="cuda", timer=None) -> dict:
    """Always feed golden z[t], including after a wrong student prediction.

    encode/predict are the fixed runner adapters. A CPU device and an untimed
    callback allow the feedback and input-immutability contract to be unit tested.
    """
    def invoke(forward):
        return (forward(), None) if timer is None else timer(forward)

    def readonly(check, changes):
        if changes:
            check["passed"] = False
            check["reason"] = "Modified read-only input: " + ", ".join(changes)
        return check

    records, timings = [], []
    pixels = case["pixels"].to(device).clone()
    before_pixels = pixels.clone()
    initial, measured = invoke(lambda: encode(model, pixels))
    if measured is not None:
        timings.append({"kind": "encode_initial", **measured})
    changed_pixels = [] if torch.equal(pixels, before_pixels) else ["pixels"]
    valid_initial = isinstance(initial, Mapping) and set(initial) == set(LEVELS)
    for level in LEVELS:
        result = (compare_tensor(initial[level], golden["initial"][level], epsilon, device)
                  if valid_initial else {"passed": False, "reason": "Expected initial states for exactly levels 1, 2, 3"})
        records.append({"kind": "initial", "level": level,
                        "validation": readonly(result, changed_pixels)})
    del initial, pixels, before_pixels

    for level in LEVELS:
        bank = case["actions"][level]
        batch, samples, horizon, action_dim = bank.shape
        actions = bank.to(device).reshape(batch * samples, horizon, action_dim)
        reference = golden["rollouts"][level]
        for step in range(horizon):
            # Never use a student's encoder or preceding prediction as context.
            latent = reference[:, :, step:step + 1].reshape(
                batch * samples, 1, LATENT_DIM
            ).to(device).clone()
            prefix = actions[:, :step + 1].clone()
            before_latent, before_prefix = latent.clone(), prefix.clone()
            prediction, measured = invoke(lambda: predict(model, level, latent, prefix))
            if measured is not None:
                timings.append({"kind": "predict_step", "level": level, "step": step,
                                **measured})
            target = reference[:, :, step + 1:step + 2].reshape(
                batch * samples, 1, LATENT_DIM
            )
            result = compare_tensor(prediction, target, epsilon, device)
            changes = [name for name, actual, before in (
                ("latent", latent, before_latent), ("actions_prefix", prefix, before_prefix)
            ) if not torch.equal(actual, before)]
            records.append({"kind": "prediction", "level": level, "step": step + 1,
                            "validation": readonly(result, changes)})
            # Even a wrong prediction is discarded before the next golden feed.
            del prediction, latent, prefix, before_latent, before_prefix
        del actions
    return {"id": case["id"], "passed": all(r["validation"]["passed"] for r in records),
            "records": records, "timings": timings,
            "wall_ms": sum(t["wall_ms"] for t in timings),
            "cuda_event_ms": sum(t["cuda_event_ms"] for t in timings)}


def save_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args) -> int:
    # Keep model imports out of helper/unit-test imports.
    from runner import ASSETS, encode_initial, load_model, predict_step, runtime_environment, source_identity

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: run check.py inside an allocated GPU job")
    torch.set_num_threads(4)
    report = {"format": "fourroom_teacher_forced_check_v1", "completed": False,
              "passed": False, "epsilon": args.epsilon, "comparison": "absolute error for every scalar",
              "relative_l2": "diagnostic only; not an acceptance bound",
              "context": "Golden latent is fed before EVERY prediction, even after a failed comparison",
              "timing_scope": "Sum of initial encode and one-step prediction forward times; teacher-forced, not free-running",
              "timing_excluded": "setup, loading, transfers, context/input copies, warmup and comparison",
              "warmup_repeats": 1, "measured_repeats": args.repeats,
              "submission": str(args.submission.resolve()), "cases": []}
    save_report(args.output, report)
    try:
        with torch.inference_mode():
            model, _, manifest = load_model(submission=args.submission, device="cuda")
            cases, goldens, provenance = load_fixtures(ASSETS / "check", manifest, args.case)
            report.update(provenance)
            report["weights_sha256"] = manifest["weights_sha256"]
            report["architecture_sha256"] = manifest["architecture_sha256"]
            report["environment"] = runtime_environment()
            report["environment"]["cpu_threads"] = torch.get_num_threads()
            report["source_sha256"] = source_identity(args.submission)
            for case in cases:
                common = {"encode": encode_initial, "predict": predict_step,
                          "epsilon": args.epsilon, "device": "cuda"}
                # First sample uses identical golden contexts, but is never timed.
                check_case(model, case, goldens[case["id"]], **common)
                torch.cuda.synchronize()
                repetitions = [check_case(model, case, goldens[case["id"]],
                                          timer=CudaTimer(), **common)
                               for _ in range(args.repeats)]
                checks = [record["validation"] for repeat in repetitions for record in repeat["records"]]
                summary = {"id": case["id"], "passed": all(r["passed"] for r in repetitions),
                           "median_ms": statistics.median(r["wall_ms"] for r in repetitions),
                           "cuda_event_median_ms": statistics.median(r["cuda_event_ms"] for r in repetitions),
                           "max_abs": max((r.get("max_abs", 0) for r in checks), default=0),
                           "max_relative_l2": max((r.get("max_relative_l2", 0) for r in checks), default=0),
                           "repetitions": repetitions}
                report["cases"].append(summary)
                save_report(args.output, report)
                print(f"{case['id']}: passed={summary['passed']} "
                      f"teacher-forced median={summary['median_ms']:.3f} ms "
                      f"max_abs={summary['max_abs']:.6g}", flush=True)
        report["completed"] = True
        report["passed"] = all(case["passed"] for case in report["cases"])
        report["sum_case_median_ms"] = sum(case["median_ms"] for case in report["cases"])
        report["full_check_repetitions"] = [
            {"repeat": repeat + 1,
             "wall_ms": sum(case["repetitions"][repeat]["wall_ms"] for case in report["cases"]),
             "cuda_event_ms": sum(case["repetitions"][repeat]["cuda_event_ms"] for case in report["cases"])}
            for repeat in range(args.repeats)
        ]
        report["median_ms"] = statistics.median(r["wall_ms"] for r in report["full_check_repetitions"])
        report["cuda_event_median_ms"] = statistics.median(
            r["cuda_event_ms"] for r in report["full_check_repetitions"]
        )
        save_report(args.output, report)
        # The judge reads the verdict off the report: a nonzero exit is a crash.
        return 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save_report(args.output, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / "hw4.py")
    parser.add_argument("--epsilon", type=positive_epsilon, default=3e-4,
                        help="Provisional absolute error bound; requires grading-GPU calibration")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", help="Run one fixture case ID")
    parser.add_argument("--output", type=Path, default=ROOT / "results/check.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
