"""Startup-cost breakdown of the check-case path, stage by stage.

Mirrors check.py: import torch, CUDA context, transformers, verify/load the assets, import
hw4.py, build_model(), validate, load the check fixtures, one warmup check and three timed
repeats. Set PROF_T0=$(date +%s.%N) before launching to also measure interpreter start-up.
Writes $PROF_OUT/startup_<tag>.json and $PROF_OUT/case_ids.txt.
"""
from __future__ import annotations

import time

_T_ENTER = time.time()
_P_ENTER = time.perf_counter()

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.setup_paths()


def factory_split(weights, architecture):
    """Time the pieces of the baseline factory.build_model (a replica; only for the split)."""
    import torch
    from model import factory as F
    from model.hjepa import HJEPA
    from model.jepa import JEPA, ProjectedPredictor
    from model.modules import Embedder
    from model.predictors import ARPredictor

    perf = time.perf_counter
    split = {}
    levels = []
    for settings in architecture["levels"]:
        level = settings["level"]
        t = perf()
        encoder = F.make_encoder(settings["encoder"])
        split[f"L{level} encoder construct (CPU init)"] = perf() - t
        t = perf()
        predictor = ProjectedPredictor(
            ARPredictor(**settings["predictor"]), F.make_mlp(settings["predictor_projector"])
        )
        split[f"L{level} predictor construct (CPU init)"] = perf() - t
        t = perf()
        levels.append(JEPA(
            encoder, predictor, Embedder(**settings["action_embed"]),
            action_encoder=F.make_action_encoder(settings["action_encoder"]),
            level=level,
            action_queue_size=settings["action_queue_size"],
            temporal_stride=settings["temporal_stride"],
            temporal_window_size=settings["temporal_window_size"],
            target_length=settings["target_length"],
        ))
        split[f"L{level} embed/action-encoder/JEPA"] = perf() - t
    device = next(iter(weights.values())).device
    t = perf()
    model = HJEPA(levels).to(device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    split["HJEPA .to(cuda)"] = perf() - t
    t = perf()
    model.load_state_dict(weights, strict=True)
    torch.cuda.synchronize()
    split["load_state_dict"] = perf() - t
    t = perf()
    model.eval().requires_grad_(False)
    split["eval/requires_grad_"] = perf() - t
    del model
    return split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=None, help="check case id (default: the first one)")
    parser.add_argument("--tag", default=os.environ.get("PROF_TAG", "run"))
    parser.add_argument("--no-factory-split", action="store_true")
    args = parser.parse_args()
    out = C.out_dir()
    perf = time.perf_counter
    stages = []
    last = [perf()]

    def mark(name, note=""):
        now = perf()
        stages.append({"stage": name, "seconds": now - last[0], "note": note})
        print(f"  {name:<44s} {now - last[0]:8.3f} s  {note}", flush=True)
        last[0] = now

    t0_env = os.environ.get("PROF_T0")
    interpreter = (_T_ENTER - float(t0_env)) if t0_env else None
    if interpreter is not None:
        print(f"  {'interpreter start (shell -> script)':<44s} {interpreter:8.3f} s", flush=True)
    mark("script preamble")

    import torch
    mark("import torch", torch.__version__)
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    mark("CUDA context init", torch.cuda.get_device_name())
    import transformers
    mark("import transformers", transformers.__version__)
    from transformers import ViTConfig, ViTModel  # noqa: F401
    mark("load transformers ViT modules")
    import runner
    import check
    mark("import runner + check")

    torch.set_num_threads(4)  # check.run does the same
    report = {"tag": args.tag, "interpreter_start_s": interpreter, "stages": stages}
    with torch.inference_mode():  # check.run loads the model under inference_mode
        runner.precision_policy()
        manifest = runner.verify_assets()
        mark("verify_assets", f"HW4_ASSETS_VERIFIED={os.environ.get('HW4_ASSETS_VERIFIED')}")
        weights_path = runner.ASSETS / "weights.pt"
        weights = torch.load(weights_path, map_location="cuda", weights_only=True)
        torch.cuda.synchronize()
        mark("torch.load weights -> GPU", f"{weights_path.stat().st_size / 2**20:.1f} MiB")
        architecture = json.loads((runner.ASSETS / "architecture.json").read_text())

        path = (C.ROOT / "hw4.py").resolve()
        sys.path[:] = [entry for entry in sys.path if entry != str(path.parent)]
        sys.path.insert(0, str(path.parent))
        runner.add_reference_paths()
        spec = importlib.util.spec_from_file_location("hw4_submission", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mark("import hw4.py + model/")
        built = module.build_model(weights, architecture)
        torch.cuda.synchronize()
        mark("build_model()")
        model = runner.validate_model(built)
        mark("validate_model")

        folder = runner.ASSETS / "check"
        sizes = {name: (folder / name).stat().st_size / 2**20 for name in ("cases.pt", "golden.pt")}
        cases, goldens, _ = check.load_fixtures(folder, manifest, args.case)
        mark("check.load_fixtures (sha256 + torch.load)",
             ", ".join(f"{k} {v:.1f} MiB" for k, v in sizes.items()))
        ids = [case["id"] for case in cases]
        if args.case is None:
            (out / "case_ids.txt").write_text("\n".join(ids) + "\n")
        case = cases[0]
        common = {"encode": runner.encode_initial, "predict": runner.predict_step,
                  "epsilon": 3e-4, "device": "cuda"}
        warm = check.check_case(model, case, goldens[case["id"]], **common)
        torch.cuda.synchronize()
        mark("check warmup run (untimed by check.py)", f"case {case['id']}")
        repeats = [check.check_case(model, case, goldens[case["id"]], timer=check.CudaTimer(), **common)
                   for _ in range(3)]
        mark("check 3 timed repeats")
        report["check"] = {
            "case": case["id"], "all_case_ids": ids, "passed": all(r["passed"] for r in repeats) and warm["passed"],
            "n_timed_calls": len(repeats[0]["timings"]),
            "repeat_wall_ms": [r["wall_ms"] for r in repeats],
            "repeat_cuda_event_ms": [r["cuda_event_ms"] for r in repeats],
            "action_bank_shapes": {str(k): list(v.shape) for k, v in case["actions"].items()},
            "pixels_shape": list(case["pixels"].shape),
        }
        # The check path ends here; the factory split below is extra work for the breakdown.
        report["python_total_s"] = perf() - _P_ENTER
        if not args.no_factory_split:
            try:
                report["factory_split"] = factory_split(weights, architecture)
            except Exception as error:  # the replica only fits the baseline factory
                report["factory_split"] = {"error": repr(error)}

    report["t0_epoch"] = float(t0_env) if t0_env else None
    report["t_enter_epoch"] = _T_ENTER
    report["t_end_epoch"] = time.time()
    report["stage_sum_s"] = sum(s["seconds"] for s in stages)
    C.save_json(out / f"startup_{args.tag}.json", report)
    print(f"startup: python-side total {report['python_total_s']:.3f} s -> {out / f'startup_{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
