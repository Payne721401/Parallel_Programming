"""Fixed FourRoom assets, student loading, and shared inference operations."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parent
# The judge's copy reads the assets off the share that ../runner.py names, rather than an
# assets/ beside this file.
ASSETS = Path(os.environ.get("HW4_ASSETS") or ROOT / "assets")
# Set by ../runner.py once preflight.py has hashed every asset on the node, so a case does not
# hash the 670 MB again on its clock.
VERIFIED = os.environ.get("HW4_ASSETS_VERIFIED") == "1"
LEVELS = (1, 2, 3)


def digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def precision_policy():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def add_reference_paths():
    """Keep the central harness and planner ahead of submission helpers."""
    # Reassert priority even when a path was already present farther down.
    # This avoids a helper named hierarchical_solver.py or stable_worldmodel
    # accidentally replacing the fixed implementation. Student Python is not
    # an OS sandbox; intentional runtime mutation is subject to source review.
    for directory in reversed((ROOT, ROOT / "fixed" / "reference")):
        path = str(directory)
        sys.path[:] = [entry for entry in sys.path if entry != path]
        sys.path.insert(0, path)


def verify_assets():
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    if manifest.get("format") != "hjepa_fourroom_assets_v1":
        raise ValueError("Unsupported FourRoom assets; restore the supplied assets directory")
    for name, entry in manifest["files"].items():
        path = (ASSETS / name).resolve()
        if not path.is_relative_to(ASSETS.resolve()):
            raise ValueError(f"Invalid asset path: {name}")
        if not path.is_file() or (not VERIFIED and digest(path) != entry["sha256"]):
            raise ValueError(f"Fixed asset SHA256 mismatch: {name}")
    manifest["weights_sha256"] = manifest["files"]["weights.pt"]["sha256"]
    manifest["architecture_sha256"] = manifest["files"]["architecture.json"]["sha256"]
    return manifest


def load_assets(device="cuda"):
    precision_policy()
    manifest = verify_assets()
    weights = torch.load(ASSETS / "weights.pt", map_location=device, weights_only=True)
    architecture = json.loads((ASSETS / "architecture.json").read_text())
    if architecture.get("format") != "hjepa_fourroom_architecture_v1":
        raise ValueError("Unsupported FourRoom architecture")
    if not isinstance(weights, dict) or not weights or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in weights.items()
    ):
        raise ValueError("weights.pt must contain only named tensors")
    return weights, architecture, manifest


def validate_model(model):
    if not isinstance(model, torch.nn.Module) or model.num_levels != 3:
        raise TypeError("build_model must return a three-level torch.nn.Module")
    for level in LEVELS:
        jepa = model.get_level(level)
        for name in ("encode", "predict", "rollout", "get_cost", "get_latent_action_queue"):
            if not callable(getattr(jepa, name, None)):
                raise TypeError(f"Level {level} must implement {name}")
        if not callable(getattr(jepa.encoder, "encode_info", None)):
            raise TypeError(f"Level {level} encoder must implement encode_info")
        for name in ("action_embed", "predictor"):
            if not callable(getattr(jepa, name, None)):
                raise TypeError(f"Level {level} must expose {name}")
    return model.eval().requires_grad_(False)


def load_submission(path, weights, architecture):
    path = Path(path or ROOT / "hw4.py").resolve(strict=True)
    directory = str(path.parent)
    sys.path[:] = [entry for entry in sys.path if entry != directory]
    sys.path.insert(0, directory)
    add_reference_paths()
    spec = importlib.util.spec_from_file_location("hw4_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_model", None)):
        raise TypeError("hw4.py must define build_model(weights, architecture)")
    return validate_model(module.build_model(weights, architecture))


def load_model(submission=None, device="cuda"):
    weights, architecture, manifest = load_assets(device)
    return load_submission(submission, weights, architecture), architecture, manifest


def encode_initial(model, pixels):
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 224, 224):
        raise ValueError("Expected normalized FP32 pixels[B,3,224,224]")
    info, result = {"pixels": pixels[:, None]}, {}
    for level in LEVELS:
        key = "pixels" if level == 1 else f"embed_{level - 1}"
        out = model.get_level(level).encode(info, key=key, chunk_temporal_inputs=level > 1)
        result[level] = out["embed_0"][:, -1:]
        info[f"embed_{level}"] = out["embed_0"]
    return result


def predict_step(model, level, latent, actions_prefix):
    """One history-size-1 transition, differentiable w.r.t. latent and actions."""
    jepa = model.get_level(level)
    action_embed = jepa.action_embed(actions_prefix)[:, -1:]
    return jepa.predict(latent, action_embed)[:, -1:]


def runtime_environment(device="cuda"):
    result = {
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "autocast": torch.is_autocast_enabled(),
    }
    if torch.device(device).type == "cuda":
        p = torch.cuda.get_device_properties(device)
        result.update(gpu=p.name, gpu_uuid=str(getattr(p, "uuid", "unavailable")),
                      vram_bytes=p.total_memory, capability=[p.major, p.minor])
    return result


def source_identity(submission=None):
    path = Path(submission or ROOT / "hw4.py").resolve()
    excluded = {"assets", "fixed", "results", "logs", "tests", "__pycache__", "build", "dist"}
    suffixes = {".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}
    sources = {}
    for directory, names, filenames in os.walk(path.parent):
        names[:] = [name for name in names if name not in excluded and not name.startswith(".")]
        for filename in sorted(filenames):
            source = Path(directory) / filename
            if source.suffix in suffixes:
                sources[str(source.relative_to(path.parent))] = digest(source)
    sources[path.name] = digest(path)
    return dict(sorted(sources.items()))
