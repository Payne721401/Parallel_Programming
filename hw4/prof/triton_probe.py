"""Can Triton kernels be used here, and what do they cost at start-up?

Measures, in one fresh process:
  1. import triton                       (the judge recompiles every .py: run with -X pycache_prefix)
  2. first call of a small fused kernel  (JIT compile; TRITON_CACHE_DIR should be a fresh folder)
  3. a second kernel's first call        (does compile time add up per kernel?)
  4. steady-state GPU time per call, against the same arithmetic in PyTorch
  5. whether the kernel can be captured in a CUDA graph and replayed with new inputs
  6. bit-equality with torch.addcmul
Usage: see the command printed in the chat; needs a GPU.
"""
from __future__ import annotations

import os
import sys
import time

t0 = time.perf_counter()
import torch  # noqa: E402

t_torch = time.perf_counter() - t0
t0 = time.perf_counter()
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

t_triton = time.perf_counter() - t0


@triton.jit
def addcmul_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b * c, mask=mask)


@triton.jit
def norm_modulate_kernel(x_ptr, shift_ptr, scale_ptr, out_ptr, D, eps, BLOCK: tl.constexpr):
    """One row per program: LayerNorm (no affine) over D, then * scale + shift.

    tl.arange needs a power-of-two length, so a row is a BLOCK >= D with the tail masked.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / D
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / D
    normed = centered / tl.sqrt(var + eps)
    scale = tl.load(scale_ptr + row * D + cols, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + row * D + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * D + cols, normed * scale + shift, mask=mask)


def timed(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def gpu_us(fn, repeat=200, replays=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        keep = [fn() for _ in range(repeat)]
    torch.cuda.synchronize()
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


def main():
    print(f"torch {torch.__version__}, triton {triton.__version__}, {torch.cuda.get_device_name()}")
    print(f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}  pycache_prefix={sys.pycache_prefix}")
    print(f"import torch {t_torch:.3f} s, import triton {t_triton:.3f} s")
    N, D = 200, 192
    torch.manual_seed(0)
    a, b, c = (torch.randn(N, D, device="cuda") for _ in range(3))
    out = torch.empty_like(a)

    def tri_addcmul():
        addcmul_kernel[(triton.cdiv(N * D, 1024),)](a, b, c, out, N * D, BLOCK=1024)
        return out

    def tri_norm():
        norm_modulate_kernel[(N,)](a, b, c, out, D, 1e-6, BLOCK=256)
        return out

    _, first = timed(tri_addcmul)
    print(f"first call, addcmul kernel (JIT compile): {first:.3f} s")
    _, second = timed(tri_norm)
    print(f"first call, norm+modulate kernel (JIT compile): {second:.3f} s")
    _, again = timed(tri_addcmul)
    print(f"later call, addcmul kernel: {again * 1e6:.0f} us wall (launch + run)")

    reference = torch.addcmul(a, b, c)
    print(f"addcmul bit-identical to torch.addcmul: {torch.equal(tri_addcmul(), reference)}")
    ref_norm = torch.nn.functional.layer_norm(a, (D,), eps=1e-6) * c + b
    print(f"norm+modulate max |diff| vs torch (3 kernels): {float((tri_norm() - ref_norm).abs().max()):.2e}")

    try:
        print(f"GPU us/call inside a CUDA graph: triton addcmul {gpu_us(tri_addcmul):.2f}, "
              f"torch.addcmul {gpu_us(lambda: torch.addcmul(a, b, c)):.2f}")
        print(f"GPU us/call inside a CUDA graph: triton norm+modulate {gpu_us(tri_norm):.2f}, "
              f"torch layer_norm * c + b {gpu_us(lambda: torch.nn.functional.layer_norm(a, (D,), eps=1e-6) * c + b):.2f}")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tri_addcmul()
        a.copy_(torch.randn_like(a))
        graph.replay()
        torch.cuda.synchronize()
        print(f"graph replay with new input matches torch: {torch.equal(out, torch.addcmul(a, b, c))}")
    except Exception as error:
        print(f"CUDA graph capture of a Triton kernel FAILED: {type(error).__name__}: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
