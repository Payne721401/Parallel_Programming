"""Which formulation of the large-K matrix products avoids cuBLAS sgemm_largek?

Each case is one product the rollout performs, Y = X @ A with X (M, K) and A (K, N):
the FeedForward output layer and the backward of its input layer (K = 1024), and the
projector's output layer and the backward of its input layer (K = 2048), for 200
candidates (L2/L3) and 50 (L1). A small-K product is included as a reference.

Every variant computes the same Y. GPU time per call is measured by capturing REPEAT
calls in one CUDA graph and timing replays with CUDA events, so CPU launch cost is
excluded. The kernel named is the longest one torch.profiler sees in a single call.
Usage: gemm_bench.py [--repeat 200] [--replays 30]      (needs a GPU; no model, no assets)
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.nn import functional as F


def variants(x, w, bias):
    """x (M, K); w (N, K) as nn.Linear stores it; bias (N,). Each returns (M, N)."""
    wt = w.t().contiguous()          # (K, N), stored transposed
    M, K = x.shape
    half, quarter = K // 2, K // 4
    pad = (-M) % 64
    xpad = F.pad(x, (0, 0, 0, pad)) if pad else x
    table = {
        "linear (reference)": lambda: F.linear(x, w, bias),
        "linear, no bias": lambda: F.linear(x, w),
        "mm(x, Wt stored)": lambda: torch.mm(x, wt),
        "addmm(b, x, Wt stored)": lambda: torch.addmm(bias, x, wt),
        "mm(W, x.t()).t()": lambda: torch.mm(w, x.t()).t(),
        "bmm, batch 1": lambda: torch.bmm(x.unsqueeze(0), wt.unsqueeze(0)).squeeze(0),
        "matmul 3-D x (M,1,K)": lambda: torch.matmul(x.unsqueeze(1), wt).squeeze(1),
        "split K in 2, add": lambda: torch.mm(x[:, :half], wt[:half]) + torch.mm(x[:, half:], wt[half:]),
        "split K in 4, add": lambda: sum(torch.mm(x[:, i * quarter:(i + 1) * quarter], wt[i * quarter:(i + 1) * quarter])
                                         for i in range(4)),
        "split M in 2, cat": lambda: torch.cat([torch.mm(x[:M // 2], wt), torch.mm(x[M // 2:], wt)]),
    }
    if pad:
        table[f"M padded to {M + pad} (pad prebuilt)"] = lambda: torch.mm(xpad, wt)[:M]
        table[f"M padded to {M + pad} (pad each call)"] = lambda: torch.mm(F.pad(x, (0, 0, 0, pad)), wt)[:M]
    return table


def gpu_us_per_call(fn, repeat, replays):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        keep = [fn() for _ in range(repeat)]
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    del keep
    return begin.elapsed_time(end) * 1000.0 / (repeat * replays)


def cpu_us_per_call(fn, repeat):
    fn()
    started = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - started) * 1e6 / repeat


def kernels(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if getattr(e, "device_time", 0) and "Memcpy" not in e.name and "Memset" not in e.name]
    if not events:
        return 0, "-"
    top = max(events, key=lambda e: e.device_time)
    return len(events), top.name[:58]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cuda = args.device.startswith("cuda")
    torch.manual_seed(0)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}, repeat {args.repeat} x {args.replays} replays")

    cases = [
        ("FF out / FF-in backward, 200 candidates", 200, 1024, 192),
        ("projector out / in backward, 200 candidates", 200, 2048, 192),
        ("FF out / FF-in backward, 50 candidates", 50, 1024, 192),
        ("projector out / in backward, 50 candidates", 50, 2048, 192),
        ("reference small K: FF in, 200 candidates", 200, 192, 1024),
        ("reference: folded attention, 200 candidates", 200, 192, 192),
    ]
    with torch.no_grad():
        for label, M, K, N in cases:
            x = torch.randn(M, K, device=args.device)
            w = torch.randn(N, K, device=args.device) / K ** 0.5
            bias = torch.randn(N, device=args.device)
            table = variants(x, w, bias)
            reference = F.linear(x, w)
            print(f"\n== {label}: X ({M}, {K}) @ A ({K}, {N}) ==")
            print(f"{'variant':34s} {'us/call':>8s} {'vs ref':>7s} {'kernels':>7s} {'max |diff|':>10s}  longest kernel")
            base = None
            for name, fn in table.items():
                try:
                    us = gpu_us_per_call(fn, args.repeat, args.replays) if cuda else cpu_us_per_call(fn, args.repeat)
                    out = fn()
                    target = reference + bias if ("b," in name or name.startswith("linear (")) else reference
                    diff = float((out - target).abs().max())
                    count, top = kernels(fn) if cuda else (0, "-")
                    base = us if base is None else base
                    print(f"{name:34s} {us:8.2f} {us / base:7.2f} {count:7d} {diff:10.2e}  {top}")
                except Exception as error:  # a variant that cannot be captured is reported, not fatal
                    print(f"{name:34s} failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
