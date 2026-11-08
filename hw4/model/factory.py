"""Construct the full editable FourRoom model from JSON and tensor-only state.

Runtime construction does not import the source checkpoint or upstream repo.
All trained parameters, BatchNorm buffers and latent-action queues are loaded
strictly. Freezing parameters retains gradients with respect to action inputs.
"""
from __future__ import annotations

from functools import partial

import torch
from torch import nn
from transformers import ViTConfig, ViTModel

from .hjepa import HJEPA
from .jepa import JEPA, ProjectedEncoder, ProjectedPredictor
from .modules import Embedder, MLP, ResidualLatentMLP
from .predictors import ARPredictor
from .sequence import SequenceEncoder


def make_mlp(settings):
    settings = dict(settings)
    normalization = settings.pop("normalization")
    eps = settings.pop("normalization_eps", 1e-5)
    norm_fn = {
        "batchnorm": partial(nn.BatchNorm1d, eps=eps),
        "layernorm": partial(nn.LayerNorm, eps=eps),
        "identity": None,
    }[normalization]
    return MLP(**settings, norm_fn=norm_fn)


def make_encoder(settings):
    if settings["kind"] == "vit":
        config = ViTConfig.from_dict(settings["vit_config"])
        config._attn_implementation = settings["attention_implementation"]
        encoder = ViTModel(
            config,
            add_pooling_layer=settings["add_pooling_layer"],
            use_mask_token=settings["use_mask_token"],
        )
        return ProjectedEncoder(encoder, make_mlp(settings["projector"]))
    if settings["kind"] == "latent_mlp":
        return ProjectedEncoder(ResidualLatentMLP(**settings["mlp"]))
    raise ValueError(f"Unsupported encoder kind: {settings['kind']}")


def make_action_encoder(settings):
    if settings["kind"] == "identity":
        return nn.Identity()
    if settings["kind"] == "sequence":
        return SequenceEncoder(**settings["settings"])
    raise ValueError(f"Unsupported action encoder kind: {settings['kind']}")


def build_model(weights, architecture):
    """Return an eval-mode HJEPA with differentiable rollout/cost methods."""
    if architecture.get("format") != "hjepa_fourroom_architecture_v1":
        raise ValueError("Unsupported FourRoom architecture")
    if not weights or not all(isinstance(v, torch.Tensor) for v in weights.values()):
        raise ValueError("Expected a nonempty tensor-only weight dictionary")
    devices = {v.device for v in weights.values()}
    if len(devices) != 1:
        raise ValueError("All weight tensors must be on the same device")
    levels = []
    for settings in architecture["levels"]:
        levels.append(JEPA(
            make_encoder(settings["encoder"]),
            ProjectedPredictor(
                ARPredictor(**settings["predictor"]),
                make_mlp(settings["predictor_projector"]),
            ),
            Embedder(**settings["action_embed"]),
            action_encoder=make_action_encoder(settings["action_encoder"]),
            level=settings["level"],
            action_queue_size=settings["action_queue_size"],
            temporal_stride=settings["temporal_stride"],
            temporal_window_size=settings["temporal_window_size"],
            target_length=settings["target_length"],
        ))
    if len(levels) != architecture["num_levels"]:
        raise ValueError("Architecture level count does not match its metadata")
    model = HJEPA(levels).to(device=next(iter(devices)), dtype=torch.float32)
    model.load_state_dict(weights, strict=True)
    model.embed_dims = {int(k): int(v) for k, v in architecture["embed_dims"].items()}
    model.pixel_embed_dims = {}
    model.proprio_embed_dims = {}
    model.interpolate_pos_encoding = True
    return model.eval().requires_grad_(False)
