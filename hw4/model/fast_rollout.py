"""Batched single-token rollout: the same mathematics as JEPA._rollout_core, fewer kernels.

With one history token the predictor's inputs split into two parts:
  * everything that depends only on the actions (action embedding, the predictor's
    condition projection, each block's adaLN shift/scale/gate) is known for all T
    steps before the rollout starts, so it is computed once for all steps instead of
    once per step on a growing prefix;
  * the latent path keeps every operation of the reference (both LayerNorms, the V
    and output projections, the feed-forward, the final norm and the projector), on
    2-D tensors so each Linear is one addmm, with modulate/gate/residual as addcmul.
Single-token attention is softmax over one score, so its output is the V projection.

No weight is changed or approximated; only the order of evaluation and of some
floating-point additions differs. HW4_FAST_ROLLOUT=0 restores the reference loop.
"""
from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn import functional as F

from .modules import Attention, ConditionalBlock, FeedForward, Transformer
from .predictors import ARPredictor

# On by default: 49/50 tasks succeeded on Nano5 with it. HW4_FAST_ROLLOUT=0 disables it.
ENABLED = os.environ.get("HW4_FAST_ROLLOUT", "1") != "0"


def supported(jepa, history: int, history_size: int) -> bool:
    """True when jepa's predictor has exactly the structure this path evaluates."""
    if not ENABLED or history != 1 or history_size != 1 or jepa.training:
        return False
    wrapper = jepa.predictor
    predictor = getattr(wrapper, "predictor", None)
    if type(predictor) is not ARPredictor or type(predictor.transformer) is not Transformer:
        return False
    for block in predictor.transformer.layers:
        if type(block) is not ConditionalBlock or type(block.attn) is not Attention:
            return False
        if type(block.mlp) is not FeedForward or block.attn.to_qkv.bias is not None:
            return False
    return True


# The attention's V projection (no bias) feeds its output projection directly when there is
# one token, so the two matrices are multiplied once at build time: to_out(V x) = (W_out W_v) x + b.
FOLD = os.environ.get("HW4_FOLD_VO", "1") != "0"
_FOLDED = "_hw4_vo_weight"


@torch.no_grad()
def prepare(jepa) -> None:
    """Precompute each block's folded V/output matrix. Call after the weights are loaded."""
    for block in jepa.predictor.predictor.transformer.layers:
        attn = block.attn
        attn.__dict__.pop(_FOLDED, None)
        if not (FOLD and supported(jepa, 1, 1)):
            continue
        out = attn.to_out[0] if isinstance(attn.to_out, nn.Sequential) else None
        if not isinstance(out, nn.Linear):
            continue
        inner_dim = attn.to_qkv.out_features // 3
        w_v = attn.to_qkv.weight[2 * inner_dim:]
        # The product is formed in float64 and rounded once to float32.
        attn.__dict__[_FOLDED] = (out.weight.double() @ w_v.double()).to(out.weight.dtype).contiguous()


def rollout(jepa, emb0: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
    """emb0 (B, 1, D), action_sequence (B, S, T, A) -> predicted stream (B, S, T + 1, D)."""
    B, S, T = action_sequence.shape[:3]
    N = B * S
    predictor = jepa.predictor.predictor
    projector = jepa.predictor.projector
    transformer = predictor.transformer

    # Action-only part, all T steps at once: (N, T, hidden).
    act_emb = jepa.action_embed(action_sequence.reshape(N, T, *action_sequence.shape[3:]))
    cond = transformer.cond_proj(act_emb)
    # conditions[b][k][t]: block b, term k (shift/scale/gate for attention, then for the MLP),
    # step t, each (N, hidden). unbind splits each (N, T, hidden) term once, so its backward
    # is one stack per term; indexing [:, t] instead would make backward zero-fill a full
    # (N, T, hidden) gradient for every step and add them up.
    conditions = []
    for block in transformer.layers:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        terms = (shift_msa, 1 + scale_msa, gate_msa, shift_mlp, 1 + scale_mlp, gate_mlp)
        conditions.append([term.unbind(dim=1) for term in terms])

    position = predictor.pos_embedding[0, :1]  # (1, D)
    emb = emb0.unsqueeze(1).expand(B, S, -1, -1).reshape(N, -1)  # (N, D); history is 1
    stream = [emb]
    for t in range(T):
        x = transformer.input_proj(emb + position)
        for block, (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) in zip(
                transformer.layers, conditions):
            attn = block.attn
            inner_dim = attn.to_qkv.out_features // 3
            h = torch.addcmul(shift_msa[t], block.norm1(x), scale_msa[t])
            folded = attn.__dict__.get(_FOLDED)
            if folded is not None:
                h = F.linear(attn.norm(h), folded, attn.to_out[0].bias)
            else:
                h = attn.to_out(F.linear(attn.norm(h), attn.to_qkv.weight[2 * inner_dim:]))
            x = torch.addcmul(x, gate_msa[t], h)
            h = torch.addcmul(shift_mlp[t], block.norm2(x), scale_mlp[t])
            x = torch.addcmul(x, gate_mlp[t], block.mlp(h))
        emb = projector(transformer.output_proj(transformer.norm(x)))
        stream.append(emb)
    return torch.stack(stream, dim=1).view(B, S, T + 1, -1)
