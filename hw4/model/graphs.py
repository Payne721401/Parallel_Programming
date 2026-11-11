"""CUDA Graph replay of JEPA.rollout inside the planner's gradient-descent loop.

Each planner iteration runs rollout -> cost -> backward with the same shapes for
30-90 iterations per solve. Eager PyTorch launches ~2,000 small kernels per
iteration, one by one, from the CPU. Here every (shape, numerics setting) gets one
forward and one backward CUDA graph, captured with
torch.cuda.make_graphed_callables from the same JEPA._rollout_core code the eager
path runs, and later calls replay them. The kernels, their inputs and their order
are the same as eager, so the cost and the gradient with respect to the actions
are bit-identical; only the CPU launch work disappears.

A graph is used only when the call differentiates through the actions. Everything
else runs eager: check.py (inference_mode), the planner's no-grad final rollouts,
and any shape whose capture failed. Every call recomputes from its own inputs;
nothing but the launch schedule is reused.

HW4_ROLLOUT_GRAPH=0 disables graphs. HW4_GRAPH_LOG=1 prints one line per capture.
"""
from __future__ import annotations

import functools
import os
import sys
import time

import torch

ENABLED = os.environ.get("HW4_ROLLOUT_GRAPH", "1") != "0"
LOG = os.environ.get("HW4_GRAPH_LOG", "0") == "1"

_CACHE_ATTR = "_hw4_rollout_graphs"
_FAILED = "capture-failed"
_SDP_FLAGS = ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled")


def _numerics_key() -> tuple:
    """Global switches that change which kernels eager mode would launch."""
    cuda = torch.backends.cuda
    sdp = tuple(getattr(cuda, name)() if hasattr(cuda, name) else None for name in _SDP_FLAGS)
    return (
        sdp,
        cuda.matmul.allow_tf32,
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )


def rollout_graph(jepa, emb0: torch.Tensor, actions: torch.Tensor, history: int, history_size: int):
    """Return a graphed callable (emb0, actions) -> predicted stream, or None to run eager."""
    if not ENABLED or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        return None
    if not actions.requires_grad or emb0.requires_grad or jepa.training:
        return None
    if not (actions.is_cuda and emb0.is_cuda) or torch.cuda.is_current_stream_capturing():
        return None

    key = (
        tuple(emb0.shape), emb0.dtype,
        tuple(actions.shape), actions.dtype, actions.device,
        int(history), int(history_size),
        _numerics_key(),
    )
    cache = jepa.__dict__.setdefault(_CACHE_ATTR, {})
    graphed = cache.get(key)
    if graphed is None:
        graphed = _capture(jepa, emb0, actions, history, history_size)
        cache[key] = graphed
    return None if graphed is _FAILED else graphed


def _capture(jepa, emb0, actions, history, history_size):
    core = functools.partial(jepa._rollout_core, history=int(history), history_size=int(history_size))
    # Private sample tensors become the graphs' static inputs; the caller's tensors are never written.
    sample = (emb0.detach().clone(), actions.detach().clone().requires_grad_(True))
    device = actions.device
    started = time.perf_counter()
    before = torch.cuda.memory_allocated(device) if LOG else 0
    try:
        graphed = torch.cuda.make_graphed_callables(core, sample)
    except Exception as error:  # this shape stays eager
        print(f"[hw4-graph] capture failed for level {jepa.level} actions {tuple(actions.shape)}, "
              f"running eager: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return _FAILED
    if LOG:
        torch.cuda.synchronize(device)
        grown = (torch.cuda.memory_allocated(device) - before) / 2**20
        print(f"[hw4-graph] level {jepa.level} actions {tuple(actions.shape)} captured in "
              f"{time.perf_counter() - started:.3f} s, +{grown:.0f} MiB allocated",
              file=sys.stderr, flush=True)
    return graphed
