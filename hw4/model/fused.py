"""Triton kernels that fuse the element-wise chain around each predictor GEMM.

One kernel evaluates, per candidate row (D = hidden size):
    u = x + g * (o + b)          residual: previous branch output o (bias b) times gate g
    n = LayerNorm(u)             no affine, eps1
    h = shift + n * scale        adaLN modulate (scale already holds 1 + scale)
    a = LayerNorm(h) * gamma + beta      eps2
and a second kernel evaluates the whole backward of that chain. Compile-time switches
drop the residual (first block of a step) or the modulate (end of a step, where the
transformer's final LayerNorm follows the residual directly; then a = n * gamma + beta).

Every operation of the reference is kept; only kernel boundaries change, so results
differ from PyTorch in the last bits (FMA contraction). Enabled with HW4_FUSED=1.
Triton is imported on first use, so check.py never loads it.
"""
from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("HW4_FUSED", "0") == "1"
BLOCK = 256  # tl.arange needs a power of two; rows are D <= BLOCK wide with the tail masked
_K = {}


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def chain_fwd(x_ptr, g_ptr, o_ptr, b_ptr, shift_ptr, scale_ptr, gamma_ptr, beta_ptr, u_ptr, a_ptr,
                  D, g_stride, shift_stride, scale_stride, inv_d, eps1, eps2,
                  HAS_RES: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_MOD: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        u = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
        if HAS_RES:
            o = tl.load(o_ptr + row * D + cols, mask=mask, other=0.0)
            if HAS_BIAS:
                o = o + tl.load(b_ptr + cols, mask=mask, other=0.0)
            u = u + tl.load(g_ptr + row * g_stride + cols, mask=mask, other=0.0) * o
            tl.store(u_ptr + row * D + cols, u, mask=mask)
        mean = tl.sum(u, axis=0) * inv_d
        c = tl.where(mask, u - mean, 0.0)
        n = c / tl.sqrt(tl.sum(c * c, axis=0) * inv_d + eps1)
        if HAS_MOD:
            h = tl.load(shift_ptr + row * shift_stride + cols, mask=mask, other=0.0) \
                + n * tl.load(scale_ptr + row * scale_stride + cols, mask=mask, other=0.0)
            mean2 = tl.sum(h, axis=0) * inv_d
            c2 = tl.where(mask, h - mean2, 0.0)
            n = c2 / tl.sqrt(tl.sum(c2 * c2, axis=0) * inv_d + eps2)
        a = n * tl.load(gamma_ptr + cols, mask=mask, other=0.0) + tl.load(beta_ptr + cols, mask=mask, other=0.0)
        tl.store(a_ptr + row * D + cols, a, mask=mask)

    @triton.jit
    def chain_bwd(x_ptr, g_ptr, o_ptr, b_ptr, shift_ptr, scale_ptr, gamma_ptr, du_ptr, da_ptr,
                  dx_ptr, dg_ptr, do_ptr, dshift_ptr, dscale_ptr,
                  D, g_stride, shift_stride, scale_stride, inv_d, eps1, eps2,
                  HAS_RES: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_MOD: tl.constexpr, HAS_DU: tl.constexpr,
                  BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        u = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
        if HAS_RES:
            o = tl.load(o_ptr + row * D + cols, mask=mask, other=0.0)
            if HAS_BIAS:
                o = o + tl.load(b_ptr + cols, mask=mask, other=0.0)
            g = tl.load(g_ptr + row * g_stride + cols, mask=mask, other=0.0)
            u = u + g * o
        mean = tl.sum(u, axis=0) * inv_d
        c = tl.where(mask, u - mean, 0.0)
        rstd1 = 1.0 / tl.sqrt(tl.sum(c * c, axis=0) * inv_d + eps1)
        n = c * rstd1
        dn = tl.load(da_ptr + row * D + cols, mask=mask, other=0.0) * tl.load(gamma_ptr + cols, mask=mask, other=0.0)
        if HAS_MOD:
            scale = tl.load(scale_ptr + row * scale_stride + cols, mask=mask, other=0.0)
            h = tl.load(shift_ptr + row * shift_stride + cols, mask=mask, other=0.0) + n * scale
            mean2 = tl.sum(h, axis=0) * inv_d
            c2 = tl.where(mask, h - mean2, 0.0)
            rstd2 = 1.0 / tl.sqrt(tl.sum(c2 * c2, axis=0) * inv_d + eps2)
            y = c2 * rstd2
            dh = (dn - tl.sum(dn, axis=0) * inv_d - y * (tl.sum(dn * y, axis=0) * inv_d)) * rstd2
            dh = tl.where(mask, dh, 0.0)
            tl.store(dshift_ptr + row * D + cols, dh, mask=mask)
            tl.store(dscale_ptr + row * D + cols, dh * n, mask=mask)
            dn = dh * scale
        du = (dn - tl.sum(dn, axis=0) * inv_d - n * (tl.sum(dn * n, axis=0) * inv_d)) * rstd1
        if HAS_DU:
            du = du + tl.load(du_ptr + row * D + cols, mask=mask, other=0.0)
        tl.store(dx_ptr + row * D + cols, du, mask=mask)
        if HAS_RES:
            tl.store(dg_ptr + row * D + cols, du * o, mask=mask)
            tl.store(do_ptr + row * D + cols, du * g, mask=mask)

    _K.update(fwd=chain_fwd, bwd=chain_bwd)
    return _K


def _rows(t):
    """A 2-D tensor whose rows are contiguous; returns (tensor, row stride)."""
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t, t.stride(0)


def _fwd(x, g, o, bias, shift, scale, gamma, beta, eps1, eps2, has_res, has_mod):
    N, D = x.shape
    x = x.contiguous()
    a = torch.empty_like(x)
    u = torch.empty_like(x) if has_res else x
    g, gs = _rows(g) if has_res else (x, D)
    o = o.contiguous() if has_res else x
    shift, ss = _rows(shift) if has_mod else (x, D)
    scale, cs = _rows(scale) if has_mod else (x, D)
    _kernels()["fwd"][(N,)](x, g, o, bias if bias is not None else gamma, shift, scale, gamma, beta, u, a,
                            D, gs, ss, cs, 1.0 / D, eps1, eps2,
                            HAS_RES=has_res, HAS_BIAS=bias is not None, HAS_MOD=has_mod, BLOCK=BLOCK)
    return u, a


def _bwd(x, g, o, bias, shift, scale, gamma, du, da, eps1, eps2, has_res, has_mod):
    N, D = x.shape
    x = x.contiguous()
    da = da.contiguous()
    dx = torch.empty_like(x)
    dg = torch.empty_like(x) if has_res else x
    do = torch.empty_like(x) if has_res else x
    dshift = torch.empty_like(x) if has_mod else x
    dscale = torch.empty_like(x) if has_mod else x
    g, gs = _rows(g) if has_res else (x, D)
    o = o.contiguous() if has_res else x
    shift, ss = _rows(shift) if has_mod else (x, D)
    scale, cs = _rows(scale) if has_mod else (x, D)
    _kernels()["bwd"][(N,)](x, g, o, bias if bias is not None else gamma, shift, scale, gamma,
                            du.contiguous() if du is not None else x, da,
                            dx, dg, do, dshift, dscale,
                            D, gs, ss, cs, 1.0 / D, eps1, eps2,
                            HAS_RES=has_res, HAS_BIAS=bias is not None, HAS_MOD=has_mod, HAS_DU=du is not None,
                            BLOCK=BLOCK)
    return dx, dg, do, dshift, dscale


class _ModNorm(torch.autograd.Function):
    """a = LN(shift + LN0(x) * scale) * gamma + beta   (first block of a step)."""

    @staticmethod
    def forward(ctx, x, shift, scale, gamma, beta, eps1, eps2):
        ctx.save_for_backward(x, shift, scale, gamma)
        ctx.eps = (eps1, eps2)
        return _fwd(x, None, None, None, shift, scale, gamma, beta, eps1, eps2, False, True)[1]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, da):
        x, shift, scale, gamma = ctx.saved_tensors
        dx, _, _, dshift, dscale = _bwd(x, None, None, None, shift, scale, gamma, None, da, *ctx.eps, False, True)
        return dx, dshift, dscale, None, None, None, None


class _ResModNorm(torch.autograd.Function):
    """u = x + g * (o + bias);  a = LN(shift + LN0(u) * scale) * gamma + beta.  Returns (u, a)."""

    @staticmethod
    def forward(ctx, x, g, o, bias, shift, scale, gamma, beta, eps1, eps2):
        ctx.save_for_backward(x, g, o, bias, shift, scale, gamma)
        ctx.eps = (eps1, eps2)
        return _fwd(x, g, o, bias, shift, scale, gamma, beta, eps1, eps2, True, True)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, du, da):
        x, g, o, bias, shift, scale, gamma = ctx.saved_tensors
        dx, dg, do, dshift, dscale = _bwd(x, g, o, bias, shift, scale, gamma, du, da, *ctx.eps, True, True)
        return dx, dg, do, None, dshift, dscale, None, None, None, None


class _ResNorm(torch.autograd.Function):
    """a = LN(x + g * (o + bias)) * gamma + beta   (end of a step; eps is the LayerNorm's)."""

    @staticmethod
    def forward(ctx, x, g, o, bias, gamma, beta, eps):
        ctx.save_for_backward(x, g, o, bias, gamma)
        ctx.eps = eps
        return _fwd(x, g, o, bias, None, None, gamma, beta, eps, eps, True, False)[1]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, da):
        x, g, o, bias, gamma = ctx.saved_tensors
        dx, dg, do, _, _ = _bwd(x, g, o, bias, None, None, gamma, None, da, ctx.eps, ctx.eps, True, False)
        return dx, dg, do, None, None, None, None


mod_norm = _ModNorm.apply
res_mod_norm = _ResModNorm.apply
res_norm = _ResNorm.apply
