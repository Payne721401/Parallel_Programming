# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0; see LICENSE.transformers.
# Modified for HW4: PyTorch-only modules, strict checkpoint names, and a reduced
# forward interface. Reference: transformers v4.57.1 and v5.21.0 modeling_vit.py.
"""PyTorch-only ViT with the released Hugging Face parameter hierarchy.

Matches the operations used by transformers ViTModel; removes its import and
generic model-wrapper overhead. Checkpoint tensors are still loaded strictly by
factory.py. Q/K/V projections, attention backend selection, both layer norms,
GELU, CLS/mask/position parameters and optional pooling are retained.

Implementation reference: Hugging Face transformers models/vit/modeling_vit.py
(Apache-2.0). This module only implements the model interface used by HW4, with
explicit errors for unsupported configurations instead of silent approximation.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class ViTOutput(NamedTuple):
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor | None


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def _config(values, attention_implementation):
    defaults = dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                    intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.0,
                    attention_probs_dropout_prob=0.0, layer_norm_eps=1e-12,
                    image_size=224, patch_size=16, num_channels=3, qkv_bias=True,
                    pooler_output_size=None, pooler_act="tanh", return_dict=True)
    config = SimpleNamespace(**{**defaults, **dict(values)})
    if config.hidden_act != "gelu":
        raise ValueError("NativeViT currently requires the original exact GELU activation")
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("ViT hidden_size must be divisible by num_attention_heads")
    if getattr(config, "pruned_heads", None):
        raise ValueError("NativeViT does not support pruned attention heads")
    if attention_implementation not in ("sdpa", "eager"):
        raise ValueError(f"Unsupported ViT attention backend: {attention_implementation}")
    config._attn_implementation = attention_implementation
    return config


class _PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = _pair(config.image_size)
        self.patch_size = _pair(config.patch_size)
        self.num_channels = config.num_channels
        self.num_patches = ((self.image_size[0] // self.patch_size[0]) *
                            (self.image_size[1] // self.patch_size[1]))
        self.projection = nn.Conv2d(config.num_channels, config.hidden_size,
                                    kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, pixels, interpolate_pos_encoding=False):
        if pixels.shape[1] != self.num_channels:
            raise ValueError("ViT input channel count does not match the checkpoint")
        if not interpolate_pos_encoding and tuple(pixels.shape[-2:]) != self.image_size:
            raise ValueError("ViT image size differs; positional interpolation is required")
        return self.projection(pixels).flatten(2).transpose(1, 2)


class _Embeddings(nn.Module):
    def __init__(self, config, use_mask_token):
        super().__init__()
        # All values are replaced by strict checkpoint loading before evaluation.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = _PatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embeddings.num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = _pair(config.patch_size)

    def interpolate_pos_encoding(self, tokens, height, width):
        positions = self.position_embeddings
        if (not torch.jit.is_tracing() and tokens.shape[1] == positions.shape[1]
                and height == width):
            return positions
        count = positions.shape[1] - 1
        side = int(count ** 0.5)
        if side * side != count:
            raise ValueError("ViT position interpolation requires a square checkpoint grid")
        patches = positions[:, 1:].reshape(1, side, side, tokens.shape[-1]).permute(0, 3, 1, 2)
        patches = F.interpolate(patches, size=(height // self.patch_size[0], width // self.patch_size[1]),
                                mode="bicubic", align_corners=False)
        patches = patches.permute(0, 2, 3, 1).view(1, -1, tokens.shape[-1])
        return torch.cat((positions[:, :1], patches), dim=1)

    def forward(self, pixels, bool_masked_pos=None, interpolate_pos_encoding=False):
        tokens = self.patch_embeddings(pixels, interpolate_pos_encoding)
        if bool_masked_pos is not None:
            if self.mask_token is None:
                raise ValueError("Checkpoint has no mask_token")
            replacement = self.mask_token.expand(pixels.shape[0], tokens.shape[1], -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(replacement)
            tokens = tokens * (1.0 - mask) + replacement * mask
        tokens = torch.cat((self.cls_token.expand(pixels.shape[0], -1, -1), tokens), dim=1)
        positions = (self.interpolate_pos_encoding(tokens, *pixels.shape[-2:])
                     if interpolate_pos_encoding else self.position_embeddings)
        return self.dropout(tokens + positions)


class _SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        self.scaling = self.attention_head_size ** -0.5
        self.dropout_prob = config.attention_probs_dropout_prob
        self.implementation = config._attn_implementation
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)

    def forward(self, hidden):
        shape = (hidden.shape[0], -1, self.num_attention_heads, self.attention_head_size)
        k = self.key(hidden).view(*shape).transpose(1, 2)
        v = self.value(hidden).view(*shape).transpose(1, 2)
        q = self.query(hidden).view(*shape).transpose(1, 2)
        drop = self.dropout_prob if self.training else 0.0
        if self.implementation == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                dropout_p=drop, is_causal=False, scale=self.scaling)
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            probabilities = F.dropout(probabilities, p=drop, training=self.training)
            out = torch.matmul(probabilities, v)
        return out.transpose(1, 2).contiguous().reshape(hidden.shape[0], -1, self.all_head_size)


class _SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden):
        return self.dropout(self.dense(hidden))


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _SelfAttention(config)
        self.output = _SelfOutput(config)

    def forward(self, hidden):
        return self.output(self.attention(hidden))


class _Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU(approximate="none")

    def forward(self, hidden):
        return self.intermediate_act_fn(self.dense(hidden))


class _Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden, residual):
        return self.dropout(self.dense(hidden)) + residual


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _Attention(config)
        self.intermediate = _Intermediate(config)
        self.output = _Output(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden):
        hidden = self.attention(self.layernorm_before(hidden)) + hidden
        return self.output(self.intermediate(self.layernorm_after(hidden)), hidden)


class _Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([_Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden):
        for layer in self.layer:
            hidden = layer(hidden)
        return hidden


class _Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.pooler_act != "tanh":
            raise ValueError("NativeViT currently requires tanh pooling")
        self.dense = nn.Linear(config.hidden_size, config.pooler_output_size or config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden):
        return self.activation(self.dense(hidden[:, 0]))


class _AttentionV5(nn.Module):
    """transformers 5.x ViTAttention: q_proj/k_proj/v_proj/o_proj, same operations."""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_prob = config.attention_probs_dropout_prob
        self.implementation = config._attn_implementation
        inner = self.num_attention_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, inner, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, inner, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, inner, bias=config.qkv_bias)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=True)

    def forward(self, hidden):
        input_shape = hidden.shape[:-1]
        shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden).view(shape).transpose(1, 2)
        k = self.k_proj(hidden).view(shape).transpose(1, 2)
        v = self.v_proj(hidden).view(shape).transpose(1, 2)
        drop = self.dropout_prob if self.training else 0.0
        if self.implementation == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drop,
                                                scale=self.scaling, is_causal=False)
        else:
            weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
            weights = F.dropout(weights, p=drop, training=self.training)
            out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous()
        return self.o_proj(out.reshape(*input_shape, -1).contiguous())


class _MLPV5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden):
        return self.fc2(F.gelu(self.fc1(hidden)))


class _LayerV5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _AttentionV5(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _MLPV5(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden):
        hidden = self.dropout(self.attention(self.layernorm_before(hidden))) + hidden
        return self.dropout(self.mlp(self.layernorm_after(hidden))) + hidden


class NativeViT(nn.Module):
    """Checkpoint-compatible ViT for pixel_values -> last_hidden_state/pooler_output.

    layout "v5" uses the transformers 5.x parameter names (layers.N.attention.q_proj,
    layers.N.mlp.fc1, ...); "v4" uses the 4.x names (encoder.layer.N.attention.attention.query, ...).
    """

    def __init__(self, config, *, attention_implementation="sdpa", add_pooling_layer=True,
                 use_mask_token=False, layout="v5"):
        super().__init__()
        if layout not in ("v4", "v5"):
            raise ValueError(f"Unknown ViT parameter layout: {layout}")
        self.layout = layout
        self.config = _config(config, attention_implementation)
        self.embeddings = _Embeddings(self.config, use_mask_token)
        if layout == "v5":
            self.layers = nn.ModuleList([_LayerV5(self.config) for _ in range(self.config.num_hidden_layers)])
        else:
            self.encoder = _Encoder(self.config)
        self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.pooler = _Pooler(self.config) if add_pooling_layer else None

    def forward(self, pixel_values, bool_masked_pos=None, head_mask=None,
                interpolate_pos_encoding=False, return_dict=None):
        if head_mask is not None:
            raise ValueError("NativeViT does not support head_mask")
        pixels = pixel_values.to(dtype=self.embeddings.patch_embeddings.projection.weight.dtype)
        hidden = self.embeddings(pixels, bool_masked_pos, interpolate_pos_encoding)
        if self.layout == "v5":
            for layer in self.layers:
                hidden = layer(hidden)
        else:
            hidden = self.encoder(hidden)
        hidden = self.layernorm(hidden)
        pooled = self.pooler(hidden) if self.pooler is not None else None
        if return_dict is False or (return_dict is None and not self.config.return_dict):
            return (hidden,) if pooled is None else (hidden, pooled)
        return ViTOutput(hidden, pooled)
