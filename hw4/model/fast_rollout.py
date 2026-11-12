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

from . import fused
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


class _AddCMul(torch.autograd.Function):
    """a + b * c with the backward written out.

    torch.addcmul's derivative is grad * (c * value) even for value == 1, two kernels
    per product. Returning grad * c and grad * b gives the same bits with one each.
    """

    @staticmethod
    def forward(ctx, a, b, c):
        ctx.save_for_backward(b, c)
        return torch.addcmul(a, b, c)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        b, c = ctx.saved_tensors
        need_a, need_b, need_c = ctx.needs_input_grad
        return (grad if need_a else None,
                grad * c if need_b else None,
                grad * b if need_c else None)


_addcmul = _AddCMul.apply


# The attention's V projection (no bias) feeds its output projection directly when there is
# one token, so the two matrices are multiplied once at build time: to_out(V x) = (W_out W_v) x + b.
FOLD = os.environ.get("HW4_FOLD_VO", "1") != "0"
_FOLDED = "_hw4_vo_weight"


# cuBLAS picks sgemm_largek (about 25 us on an H100) for Linear layers whose input is wide
# (K = 1024 or 2048) when they run as addmm. The same product as mm against a weight stored
# transposed takes about 9 us, so those layers keep a transposed copy and add the bias after.
WIDE_MM = os.environ.get("HW4_WIDE_MM", "1") != "0"
WIDE_K = 512
_TRANSPOSED = "_hw4_weight_t"


def _run(sequential, x):
    """sequential(x) in eval mode, with wide-input Linear layers as mm + bias."""
    for module in sequential:
        if isinstance(module, nn.Dropout):
            continue
        weight_t = module.__dict__.get(_TRANSPOSED)
        if weight_t is None:
            x = module(x)
        else:
            x = torch.mm(x, weight_t)
            if module.bias is not None:
                x = x + module.bias
    return x


@torch.no_grad()
def prepare(jepa) -> None:
    """Precompute the folded V/output matrices and transposed wide weights.

    Call after the weights are loaded and the model is in eval mode.
    """
    usable = supported(jepa, 1, 1)
    projector = jepa.predictor.projector
    sequences = [block.mlp.net for block in jepa.predictor.predictor.transformer.layers]
    if isinstance(getattr(projector, "net", None), nn.Sequential):
        sequences.append(projector.net)
    for sequential in sequences:
        for module in sequential:
            module.__dict__.pop(_TRANSPOSED, None)
            if usable and WIDE_MM and isinstance(module, nn.Linear) and module.in_features >= WIDE_K:
                module.__dict__[_TRANSPOSED] = module.weight.t().contiguous()
    for block in jepa.predictor.predictor.transformer.layers:
        attn = block.attn
        attn.__dict__.pop(_FOLDED, None)
        if not (FOLD and usable):
            continue
        out = attn.to_out[0] if isinstance(attn.to_out, nn.Sequential) else None
        if not isinstance(out, nn.Linear):
            continue
        inner_dim = attn.to_qkv.out_features // 3
        w_v = attn.to_qkv.weight[2 * inner_dim:]
        # The product is formed in float64 and rounded once to float32.
        attn.__dict__[_FOLDED] = (out.weight.double() @ w_v.double()).to(out.weight.dtype).contiguous()
    jepa.__dict__[_FUSED] = _fusable(jepa)


_FUSED = "_hw4_fused"


def ensure_prepared(jepa) -> None:
    """Run prepare() once, on the first rollout, so check.py (which never rolls out) skips it."""
    if _FUSED not in jepa.__dict__:
        # Normal tensors even if the first rollout happens under inference_mode.
        with torch.inference_mode(False):
            prepare(jepa)


def _fusable(jepa) -> bool:
    """True when every piece the Triton path (model/fused.py) relies on is present."""
    if not (fused.ENABLED and supported(jepa, 1, 1)):
        return False
    transformer = jepa.predictor.predictor.transformer
    for block in transformer.layers:
        net = block.mlp.net
        kinds = (nn.LayerNorm, nn.Linear, nn.GELU, nn.Dropout, nn.Linear, nn.Dropout)
        if len(net) != 6 or not all(isinstance(m, k) for m, k in zip(net, kinds)):
            return False
        if net[2].approximate != "none" or _TRANSPOSED not in net[4].__dict__ or net[4].bias is None:
            return False
        if _FOLDED not in block.attn.__dict__ or block.attn.to_out[0].bias is None:
            return False
        if block.norm1.elementwise_affine or block.norm2.elementwise_affine:
            return False
        if not (block.attn.norm.elementwise_affine and net[0].elementwise_affine):
            return False
        if block.norm1.weight is not None or transformer.norm.weight is None:
            return False
    return transformer.norm.normalized_shape[0] <= fused.BLOCK


def _fused_step(transformer, conditions, x, t):
    """One predictor step on the latent path with the fused kernels; returns transformer.norm's output."""
    pending = None  # (gate, branch output, its bias) not yet added to x
    for block, terms in zip(transformer.layers, conditions):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (term[t] for term in terms)
        attn, net = block.attn, block.mlp.net
        if pending is None:
            a = fused.mod_norm(x, shift_msa, scale_msa, attn.norm.weight, attn.norm.bias,
                               block.norm1.eps, attn.norm.eps)
        else:
            x, a = fused.res_mod_norm(x, *pending, shift_msa, scale_msa, attn.norm.weight, attn.norm.bias,
                                      block.norm1.eps, attn.norm.eps)
        o = torch.mm(a, attn.__dict__[_FOLDED].t())
        x, a = fused.res_mod_norm(x, gate_msa, o, attn.to_out[0].bias, shift_mlp, scale_mlp,
                                  net[0].weight, net[0].bias, block.norm2.eps, net[0].eps)
        o = torch.mm(F.gelu(net[1](a)), net[4].__dict__[_TRANSPOSED])
        pending = (gate_mlp, o, net[4].bias)
    norm = transformer.norm
    return fused.res_norm(x, *pending, norm.weight, norm.bias, norm.eps)


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

    wide_projector = isinstance(getattr(projector, "net", None), nn.Sequential) and hasattr(projector, "final_norm")
    position = predictor.pos_embedding[0, :1]  # (1, D)
    emb = emb0.unsqueeze(1).expand(B, S, -1, -1).reshape(N, -1)  # (N, D); history is 1
    stream = [emb]
    use_fused = jepa.__dict__.get(_FUSED, False)
    for t in range(T):
        x = transformer.input_proj(emb + position)
        if use_fused:
            x = transformer.output_proj(_fused_step(transformer, conditions, x, t))
            emb = projector.final_norm(_run(projector.net, x)) if wide_projector else projector(x)
            stream.append(emb)
            continue
        for block, (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) in zip(
                transformer.layers, conditions):
            attn = block.attn
            inner_dim = attn.to_qkv.out_features // 3
            h = _addcmul(shift_msa[t], block.norm1(x), scale_msa[t])
            folded = attn.__dict__.get(_FOLDED)
            if folded is not None:
                h = F.linear(attn.norm(h), folded, attn.to_out[0].bias)
            else:
                h = attn.to_out(F.linear(attn.norm(h), attn.to_qkv.weight[2 * inner_dim:]))
            x = _addcmul(x, gate_msa[t], h)
            h = _addcmul(shift_mlp[t], block.norm2(x), scale_mlp[t])
            x = _addcmul(x, gate_mlp[t], _run(block.mlp.net, h))
        x = transformer.output_proj(transformer.norm(x))
        emb = projector.final_norm(_run(projector.net, x)) if wide_projector else projector(x)
        stream.append(emb)
    return torch.stack(stream, dim=1).view(B, S, T + 1, -1)
