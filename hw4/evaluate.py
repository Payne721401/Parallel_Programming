"""Run all 50 fixed FourRoom Distractors tasks with the submitted world model.

Spawned workers preserve the reference's one-task chunk protocol. Planning and
environment code are fixed; students optimize submission.build_model().
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import functools
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config/fourroom_eval.yaml"
TASK_COUNT = 50
BASE_SEED = 42
SUCCESS_THRESHOLD_PERCENT = 85.0
PROTOCOL_ID = "fourroom_hjepa3_fixed50_chunk1_parallel_seeded_v3"
_WORKER = {}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_value(value):
    """Convert reference metrics, including unsuccessful NaNs, to strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "detach"):
        return json_value(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(data), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def seed_sequence_tensor(values):
    """World's loader accepts a scalar Torch tensor, but skips NumPy scalars."""
    import torch

    sequence = torch.as_tensor(values, device="cpu")
    if sequence.dtype == torch.bool or sequence.is_floating_point() or sequence.is_complex():
        raise ValueError("Task environment seeds must be integers.")
    sequence = sequence.to(dtype=torch.int64).reshape(-1).clone()
    if sequence.numel() == 0 or bool((sequence < 0).any()):
        raise ValueError("Task environment seeds must be a nonempty nonnegative sequence.")
    return sequence


def prepare_task_payload(payload, index, normalize_seed_sequence=seed_sequence_tensor):
    """Select a task and make the live environment RNG independent of scheduling.

    Official FourRoom task tensors contain positions but no seed column or saved
    RNG/controller state. Generation seeds in task_info describe an earlier
    expert rollout and cannot restore its RNG at a selected interior frame.
    The fixed fallback below is an explicit deterministic evaluation convention.
    """
    episode = dict(payload["data"][index])
    columns = list(payload["columns"])
    if "seed" in episode:
        seed_values = episode["seed"]
        source = "provided_task_seed_column"
    else:
        seed = BASE_SEED + index
        frames = len(episode["pixels"])
        if frames < 1:
            raise ValueError("A task must contain an initial pixel frame.")
        seed_values = [seed] * frames
        source = "fixed_fallback_base42_plus_task_index"
    # The reference loader tests ep[column][0] against Tensor/ndarray. A 1-D
    # NumPy seed array yields np.int64 here and is silently skipped; a 1-D
    # Torch tensor yields an accepted 0-D Tensor. Normalize provided seeds too.
    episode["seed"] = normalize_seed_sequence(seed_values)
    seed = episode["seed"][0]
    seed = seed.item() if hasattr(seed, "item") else seed
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Task environment seed must be a nonnegative integer.")
    if "seed" not in columns:
        columns.append("seed")
    episode_ids = payload.get("episodes_idx")
    if episode_ids is None or len(episode_ids) == 0:
        episode_ids = list(range(len(payload["data"])))
    task = {**payload, "data": [episode], "columns": columns, "episodes_idx": [episode_ids[index]]}
    info = {"environment_seed": seed, "environment_seed_source": source}
    generation_tasks = payload.get("task_info", {}).get("tasks", [])
    if index < len(generation_tasks):
        info["task_generation_seed"] = generation_tasks[index].get("seed")
    return task, info


def validate_environment_seed(metrics, expected_seed):
    observed = json_value(metrics.get("seeds"))
    if observed != [expected_seed]:
        raise RuntimeError(
            f"Reference environment reset did not apply the required seed: expected {[expected_seed]}, got {observed}"
        )
    return observed[0]


def guard_world_reset(world, expected_seed):
    """Validate the seed before reset and adapt NumPy integers for Gymnasium."""
    original_reset = world.reset
    audit = {"applied": False, "calls": 0, "python_integer_seeds": None}

    @functools.wraps(original_reset)
    def seeded_reset(seed=None, options=None):
        normalized = json_value(seed)
        if (
            not isinstance(normalized, list)
            or len(normalized) != 1
            or type(normalized[0]) is not int
            or normalized[0] != expected_seed
        ):
            raise RuntimeError(
                f"Refusing unseeded or mismatched environment reset: expected {[expected_seed]}, got {normalized}"
            )
        # World's loader produces a NumPy array of seeds. Gymnasium 1.2.3
        # rejects np.int64 scalars, so pass a list of actual Python integers.
        output = original_reset(seed=normalized, options=options)
        audit.update(applied=True, calls=audit["calls"] + 1, python_integer_seeds=normalized)
        return output

    world.reset = seeded_reset
    return audit


def submission_sources(submission: Path):
    """Hash submitted sources, excluding fixed assets, environments and outputs."""
    excluded = {
        "assets", "cases", "golden", "results", "logs", "fixed", "tests",
        "__pycache__", "build", "dist", "venv", "env", "node_modules",
    }
    suffixes = {".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}
    files = {}
    for directory, names, filenames in os.walk(submission.parent):
        names[:] = [n for n in names if n not in excluded and not n.startswith(".")]
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix in suffixes:
                files[str(path.relative_to(submission.parent))] = digest(path)
    files[str(submission.relative_to(submission.parent))] = digest(submission)
    return dict(sorted(files.items()))


def evaluation_identity(submission: Path, device: str, forward_timing_enabled=True):
    import torch
    from runner import ASSETS, VERIFIED

    fixed_paths = sorted((ROOT / "fixed/reference").rglob("*.py"))
    fixed_paths += [CONFIG_PATH, Path(__file__).resolve(), ROOT / "runner.py"]
    assets = [ASSETS / name for name in (
        "weights.pt", "architecture.json", "normalizer.json", "fourroom_tasks.pt", "manifest.json"
    )]
    packages = {}
    for name in ("torch", "torchvision", "transformers", "numpy", "gymnasium", "hydra-core"):
        packages[name] = importlib.metadata.version(name)
    properties = torch.cuda.get_device_properties(device)
    return {
        "protocol": PROTOCOL_ID,
        "seed": BASE_SEED,
        "planner_seed_for_task_i": "42 + i, zero-based task index",
        "environment_seed_policy": "Preserve provided seed values; otherwise inject seed=42+i. Normalize to a CPU int64 Torch tensor, validate before World.reset and pass Python integers to Gymnasium, then require reference metrics.seeds to match. Generation metadata does not restore segment RNG/controller state.",
        "task_count": TASK_COUNT,
        "success_threshold_percent": SUCCESS_THRESHOLD_PERCENT,
        "forward_timing_enabled": bool(forward_timing_enabled),
        "submission": str(submission),
        "submission_sources": submission_sources(submission),
        "fixed_sources": {str(p.relative_to(ROOT)): digest(p) for p in fixed_paths},
        # Once preflight.py has checked every asset against the manifest, the manifest
        # stands for them, and the case does not hash them twice more on its clock.
        "assets": ({"manifest.json": digest(ASSETS / "manifest.json")} if VERIFIED
                   else {p.name: digest(p) for p in assets}),
        "python": platform.python_version(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "gpu_capability": [properties.major, properties.minor],
    }


class ForwardTimer:
    """Time encoder/action-embedding/predictor forward work, excluding backward.

    CUDA events are collected on the current stream. Nested component calls are
    counted once. Flushing bounds event memory; it adds measurement overhead to
    evaluation wall time, which is therefore reported separately.
    """

    enabled = True

    def __init__(self, model, device, max_pending=2048):
        import torch

        self.torch, self.device = torch, device
        self.max_pending = max_pending
        self.originals = []
        self.pending = []
        self.depth = 0
        self.totals = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0, "host_seconds": 0.0})
        for level in range(1, model.num_levels + 1):
            jepa = model.get_level(level)
            self._wrap(jepa, "encode", f"level{level}.encode")
            self._wrap(jepa, "rollout", f"level{level}.rollout")
            self._wrap(jepa.encoder, "encode_info", f"level{level}.encoder")
            self._wrap(jepa.encoder, "forward", f"level{level}.encoder")
            if hasattr(jepa, "action_encoder") and hasattr(jepa.action_encoder, "forward"):
                self._wrap(jepa.action_encoder, "forward", f"level{level}.action_encoder")
            self._wrap(jepa.action_embed, "forward", f"level{level}.action_embed")
            self._wrap(jepa.predictor, "forward", f"level{level}.predictor")

    def _wrap(self, owner, attribute, name):
        original = getattr(owner, attribute)
        self.originals.append((owner, attribute, original))

        @functools.wraps(original)
        def measured(*args, **kwargs):
            outer = self.depth == 0
            if outer:
                begin = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                started = time.perf_counter()
                begin.record()
            self.depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                self.depth -= 1
                if outer:
                    end.record()
                    total = self.totals[name]
                    total["calls"] += 1
                    total["host_seconds"] += time.perf_counter() - started
                    self.pending.append((name, begin, end))
                    if len(self.pending) >= self.max_pending:
                        self.flush()

        setattr(owner, attribute, measured)

    def flush(self):
        if self.pending:
            self.torch.cuda.synchronize(self.device)
            for name, begin, end in self.pending:
                self.totals[name]["gpu_ms"] += begin.elapsed_time(end)
            self.pending.clear()

    def reset(self):
        self.flush()
        self.totals.clear()

    def summary(self):
        self.flush()
        return {
            "enabled": True,
            "forward_gpu_seconds": sum(v["gpu_ms"] for v in self.totals.values()) / 1000,
            "forward_host_seconds": sum(v["host_seconds"] for v in self.totals.values()),
            "forward_calls": sum(v["calls"] for v in self.totals.values()),
            "components": dict(self.totals),
            "scope": "JEPA encode/rollout including allocation/cat/clone and direct encoder/action-encoder/action-embedding/predictor forward; excludes backward, optimizer, cost and environment",
            "measurement_note": "CUDA-event elapsed under parallel workers includes scheduling contention/wait, not exclusive GPU kernel busy time.",
        }


class NullForwardTimer:
    """Leave model methods untouched and do not create or record CUDA events."""

    enabled = False

    def reset(self):
        pass

    def summary(self):
        return {
            "enabled": False,
            "forward_gpu_seconds": None,
            "forward_host_seconds": None,
            "forward_calls": None,
            "components": {},
            "scope": "Forward timing disabled; complete evaluation wall time remains measured.",
        }


def make_forward_timer(model, device, enabled=True):
    return ForwardTimer(model, device) if enabled else NullForwardTimer()


def normalizer_process(path: Path, columns):
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    artifact = json.loads(path.read_text())
    stats = artifact["stats"]
    process = {}
    for column in columns:
        if column == "pixels":
            continue
        values = stats[column]
        mean = np.asarray(values["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(values["std"], dtype=np.float64).reshape(-1)
        processor = StandardScaler()
        processor.mean_, processor.scale_ = mean, scale
        processor.var_ = np.square(scale)
        processor.n_features_in_ = len(mean)
        processor.n_samples_seen_ = int(values.get("count", 0))
        process[column] = processor
        if column != "action":
            process[f"goal_{column}"] = processor
    return process


def _worker_init(submission: str, device: str, identity: dict, output_dir: str, forward_timing_enabled=True):
    started = time.perf_counter()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    from omegaconf import OmegaConf
    from runner import ASSETS, add_reference_paths, load_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(device)
    add_reference_paths()
    model, architecture, manifest = load_model(submission=Path(submission), device=device)
    if model.num_levels != 3:
        raise ValueError("The fixed evaluation requires the three-level FourRoom model.")
    model.interpolate_pos_encoding = True
    payload = torch.load(ASSETS / "fourroom_tasks.pt", map_location="cpu", weights_only=True)
    if len(payload["data"]) != TASK_COUNT:
        raise ValueError("Expected exactly 50 fixed FourRoom evaluation tasks.")
    config = OmegaConf.load(CONFIG_PATH)
    if (int(config.seed), int(config.eval.num_eval), float(config.success_threshold_percent)) != (
        BASE_SEED, TASK_COUNT, SUCCESS_THRESHOLD_PERCENT
    ):
        raise ValueError("Fixed evaluation seed, task count or success criterion was changed.")
    process = normalizer_process(ASSETS / "normalizer.json", list(config.dataset.keys_to_cache))
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    _WORKER.update({
        "model": model,
        "payload": payload,
        "config": config,
        "process": process,
        "device": device,
        "identity": identity,
        "output_dir": Path(output_dir),
        "timer": make_forward_timer(model, device, forward_timing_enabled),
        "forward_timing_enabled": bool(forward_timing_enabled),
        "setup_seconds": time.perf_counter() - started,
        "runtime": {
            "pid": os.getpid(),
            "hostname": platform.node(),
            "gpu": properties.name,
            "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "architecture_format": architecture.get("format"),
            "asset_format": manifest.get("format"),
        },
    })


def _evaluate_task(index: int):
    import numpy as np
    import torch
    from loguru import logger
    from omegaconf import OmegaConf
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torchvision.transforms import v2 as transforms
    import stable_worldmodel as swm
    from eval_config_utils import _build_policy_plan_config
    from hierarchical_solver import build_hierarchical_solver

    state = _WORKER
    log_path = state["output_dir"] / "episodes" / f"task-{index:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task_index": index,
        "planner_seed": BASE_SEED + index,
        "completed": False,
        "success": False,
        "environment_seed_verified": False,
        "forward_timing_enabled": state["forward_timing_enabled"],
        "identity": state["identity"],
        "runtime": state["runtime"],
        "worker_setup_seconds": state["setup_seconds"],
        "log": str(log_path),
    }
    world = None
    timer = state["timer"]
    with log_path.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
        logger.remove()
        logger.add(stream, level="INFO")
        try:
            cfg = OmegaConf.create(OmegaConf.to_container(state["config"], resolve=False))
            cfg.seed = BASE_SEED + index
            # Exactly matches reference chunk_size=1 seed derivation. A fresh
            # solver also prevents a worker's earlier tasks affecting this task.
            solver = build_hierarchical_solver(cfg, state["model"])
            image_transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=int(cfg.eval.img_size)),
            ])
            policy = swm.policy.WorldModelPolicy(
                solver=solver,
                config=_build_policy_plan_config(cfg),
                process=state["process"],
                transform={"pixels": image_transform, "goal": image_transform},
                eval_budget=int(cfg.eval.eval_budget),
            )
            # Preserve task frame/state data while supplying the missing reset
            # seed explicitly; no asset file is modified.
            task, seed_info = prepare_task_payload(state["payload"], index)
            result.update(seed_info)
            world_config = OmegaConf.to_container(cfg.world, resolve=True)
            world_config["seed"] = seed_info["environment_seed"]
            world = swm.World(**world_config, image_shape=(int(cfg.eval.img_size),) * 2)
            reset_audit = guard_world_reset(world, seed_info["environment_seed"])
            world.set_policy(policy)
            timer.reset()
            torch.cuda.synchronize(state["device"])
            torch.cuda.reset_peak_memory_stats(state["device"])
            started = time.perf_counter()
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                metrics = world.evaluate_from_dataset(
                    None,
                    episodes_idx=None,
                    start_steps=None,
                    goal_offset_steps=int(cfg.eval.goal_offset_steps),
                    eval_budget=int(cfg.eval.eval_budget),
                    callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
                    load_eval_trajs_path=task,
                    process=state["process"],
                    start_state_mode="dataset_full",
                    goal_state_mode="dataset_full",
                )
            torch.cuda.synchronize(state["device"])
            result["evaluation_wall_seconds"] = time.perf_counter() - started
            result["world_model_time"] = timer.summary()
            result["metrics"] = json_value(metrics)
            result["environment_reset_audit"] = reset_audit
            if reset_audit["applied"] is not True:
                raise RuntimeError("Reference evaluation did not apply the validated environment reset.")
            result["environment_seed_observed"] = validate_environment_seed(
                metrics, seed_info["environment_seed"]
            )
            result["environment_seed_verified"] = True
            if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
                raise RuntimeError("The submitted implementation enabled prohibited TF32.")
            successes = np.asarray(metrics["episode_successes"], dtype=bool)
            if successes.shape != (1,):
                raise ValueError("Each fixed task must produce exactly one success result.")
            records = []
            for level in sorted(solver.level_solvers):
                records.extend(solver.level_solvers[level].solve_records)
            result.update({
                "completed": True,
                "success": bool(successes[0]),
                "metrics": json_value(metrics),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(state["device"]),
                "planning_compute": records,
            })
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            result["traceback"] = traceback.format_exc()
            print(result["traceback"], flush=True)
        finally:
            if world is not None:
                world.close()
            logger.remove()
    return json_value(result)


def summarize(identity, results, workers, started, new_results, finished=False, task_count=TASK_COUNT):
    """`task_count` is how many tasks this run was given (--tasks), which the judge runs ten
    at a time; the 85% criterion is over all 50, and grading applies it across the cases."""
    ordered = [results[i] for i in sorted(results)]
    verified = [r for r in ordered if r.get("completed") is True and r.get("environment_seed_verified") is True]
    successful = sum(r.get("success") is True for r in verified)
    completed = len(verified) == task_count
    success_rate = successful * 100.0 / task_count
    forward_timing_enabled = bool(identity.get("forward_timing_enabled", True))
    timings = [
        r["world_model_time"] for r in ordered
        if "world_model_time" in r and r["world_model_time"].get("enabled", True)
    ]
    return {
        "format": "hw4_fourroom_evaluation_v1",
        "identity": identity,
        "finished": finished,
        "completed": completed,
        "passed": completed and success_rate >= SUCCESS_THRESHOLD_PERCENT,
        "episodes": task_count,
        "completed_episodes": len(verified),
        "environment_seed_verified_episodes": len(verified),
        "runtime_errors": len(ordered) - len(verified),
        "successes": successful,
        "success_rate_percent": success_rate,
        "success_threshold_percent": SUCCESS_THRESHOLD_PERCENT,
        "minimum_successes": math.ceil(SUCCESS_THRESHOLD_PERCENT * task_count / 100),
        "workers": workers,
        "new_tasks_this_invocation": new_results,
        "resumed_tasks": len(ordered) - new_results,
        "full_run_timing_available": completed and new_results == task_count,
        "forward_timing_enabled": forward_timing_enabled,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "sum_episode_wall_seconds": sum(r.get("evaluation_wall_seconds", 0) for r in ordered),
        "world_model_forward_gpu_seconds_sum": sum(t["forward_gpu_seconds"] for t in timings) if forward_timing_enabled else None,
        "world_model_forward_host_seconds_sum": sum(t["forward_host_seconds"] for t in timings) if forward_timing_enabled else None,
        "timing_note": "Aggregate per-worker forward GPU elapsed can exceed parallel wall time and includes GPU scheduling contention/wait, not exclusive kernel busy time. Backward/planning/environment are excluded from forward time but included in evaluation wall time. With --resume, invocation wall time covers only this invocation and is not a full-50 timing score.",
        "results": ordered,
    }


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / "hw4.py")
    parser.add_argument("--workers", type=int, default=8, help="Parallel episode workers (default: 8; allocate at least 8 CPUs)")
    parser.add_argument("--device", default="cuda:0", help="One allocated CUDA device shared by spawned workers")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluate-seeded-v3")
    parser.add_argument("--resume", action="store_true", help="Reuse completed tasks only when all source/asset/protocol identities match")
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(TASK_COUNT)),
                        help="Run only these zero-based task indices (default: all 50)")
    parser.add_argument("--no-forward-timing", action="store_true", help="Disable model hooks/CUDA events when measuring evaluation throughput; wall time remains measured")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if any(not 0 <= i < TASK_COUNT for i in args.tasks) or len(set(args.tasks)) != len(args.tasks):
        parser.error(f"--tasks must be distinct indices from 0 to {TASK_COUNT - 1}")
    args.submission = args.submission.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import torch
    from runner import verify_assets

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run evaluation inside your allocated GPU job.")
    torch.cuda.set_device(args.device)
    verify_assets()
    forward_timing_enabled = not args.no_forward_timing
    identity = evaluation_identity(args.submission, args.device, forward_timing_enabled=forward_timing_enabled)
    episode_dir = args.output_dir / "episodes"
    episode_dir.mkdir(exist_ok=True)
    results = {}
    existing = sorted(episode_dir.glob("task-*.json"))
    if existing and not args.resume:
        parser.error("Output directory already has episode results; use --resume or a new --output-dir")
    for path in existing:
        result = json.loads(path.read_text())
        if result.get("identity") != identity:
            parser.error(f"Cannot resume mismatched source/assets/runtime protocol: {path}")
        index = result.get("task_index")
        if not isinstance(index, int) or isinstance(index, bool) or index not in args.tasks:
            parser.error(f"Invalid task index in {path}")
        if path.name != f"task-{index:02d}.json" or result.get("planner_seed") != BASE_SEED + index:
            parser.error(f"Task filename or planner seed does not match its index: {path}")
        if result.get("completed") is True:
            if not isinstance(result.get("success"), bool):
                parser.error(f"Completed task is missing a boolean success result: {path}")
            recorded = result.get("metrics", {}).get("episode_successes")
            if recorded != [result["success"]]:
                parser.error(f"Completed task has contradictory reference metrics: {path}")
            try:
                validate_environment_seed(result.get("metrics", {}), result["environment_seed"])
            except (KeyError, RuntimeError) as error:
                parser.error(f"Cannot resume a task with an unverified environment seed: {path}: {error}")
            if result.get("environment_seed_verified") is not True or result.get("environment_seed_observed") != result["environment_seed"]:
                parser.error(f"Completed task lacks matching actual seed verification: {path}")
            reset_audit = result.get("environment_reset_audit", {})
            if reset_audit.get("applied") is not True or reset_audit.get("python_integer_seeds") != [result["environment_seed"]]:
                parser.error(f"Completed task lacks a validated Python-integer environment reset: {path}")
            results[index] = result
    pending = [i for i in args.tasks if i not in results]
    report_path = args.output_dir / "report.json"
    new_results = 0
    report = summarize(identity, results, args.workers, started, new_results, task_count=len(args.tasks))
    save_json(report_path, report)
    minimum = math.ceil(SUCCESS_THRESHOLD_PERCENT * TASK_COUNT / 100)
    print(f"FourRoom: {len(results)}/{len(args.tasks)} resumed; {len(pending)} pending; workers={args.workers}; minimum={minimum} successes", flush=True)
    if pending:
        # Spawn avoids inheriting a CUDA context from the coordinator.
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init,
                initargs=(str(args.submission), args.device, identity, str(args.output_dir), forward_timing_enabled),
            ) as pool:
                futures = {pool.submit(_evaluate_task, i): i for i in pending}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {
                            "task_index": index,
                            "planner_seed": BASE_SEED + index,
                            "completed": False,
                            "success": False,
                            "identity": identity,
                            "error": f"Worker failure: {type(error).__name__}: {error}",
                        }
                    results[index] = result
                    new_results += 1
                    save_json(episode_dir / f"task-{index:02d}.json", result)
                    report = summarize(identity, results, args.workers, started, new_results, task_count=len(args.tasks))
                    save_json(report_path, report)
                    print(f"task {index:02d}: completed={result['completed']} success={result['success']} ({len(results)}/{len(args.tasks)} recorded)", flush=True)
        except Exception as error:
            report["launcher_error"] = f"{type(error).__name__}: {error}"
            save_json(report_path, report)
            raise
    report = summarize(identity, results, args.workers, started, new_results, finished=True,
                       task_count=len(args.tasks))
    if evaluation_identity(args.submission, args.device, forward_timing_enabled=forward_timing_enabled) != identity:
        report["passed"] = False
        report["error"] = "Submitted sources or fixed assets changed during evaluation."
    report["wall_seconds_this_invocation"] = time.perf_counter() - started
    save_json(report_path, report)
    print(f"Success: {report['successes']}/{len(args.tasks)} = {report['success_rate_percent']:.1f}%; passed={report['passed']}")
    print(f"Wall time this invocation: {report['wall_seconds_this_invocation']:.3f} s")
    if report["world_model_forward_gpu_seconds_sum"] is None:
        print("World-model forward timing disabled; evaluation wall time is measured.")
    else:
        print(f"World-model forward GPU time summed across workers/tasks: {report['world_model_forward_gpu_seconds_sum']:.3f} s")
    print(f"Report: {report_path}")
    # The judge reads the verdict off the report: a nonzero exit is a crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
