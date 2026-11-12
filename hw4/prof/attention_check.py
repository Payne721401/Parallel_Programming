"""Isolated attention forward/backward, fallback and storage checks.

Run with --device cuda on Nano5; --device cpu needs only torch and einops.
CUDA additionally tests forward/backward graph replay with changing inputs.
This is a synthetic regression check, not the assignment's correctness gate.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Load the actual source without model/__init__ importing the unrelated ViT.
ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "_hw4_attention_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "model")]
sys.modules[PACKAGE] = package
M = importlib.import_module(f"{PACKAGE}.modules")

DEVICE = "cpu"
DIFFS = {"forward": 0.0, "input_grad": 0.0, "conditioning_grad": 0.0}


def evaluate(module, inputs, cotangent, mode):
    M.SINGLE_TOKEN_ATTENTION = mode
    values = [x.detach().clone().requires_grad_(True) for x in inputs]
    before = [x.detach().clone() for x in values]
    out = module(*values)
    grads = torch.autograd.grad(out, values, cotangent)
    for x, saved in zip(values, before):
        torch.testing.assert_close(x, saved, rtol=0, atol=0)
    return (out.detach(), *(g.detach() for g in grads))


def compare(actual, reference, exact=False):
    for name, a, b in zip(DIFFS, actual, reference):
        assert a.dtype == b.dtype == torch.float32
        assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        DIFFS[name] = max(DIFFS[name], float((a - b).abs().max()))
        torch.testing.assert_close(a, b, atol=0 if exact else 1e-5,
                                   rtol=0 if exact else 1e-5)


class AttentionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.previous = M.SINGLE_TOKEN_ATTENTION
        M.SINGLE_TOKEN_ATTENTION = "off"
        self.sdp = sdpa_kernel(SDPBackend.MATH)
        self.sdp.__enter__()

    def tearDown(self):
        self.sdp.__exit__(None, None, None)
        M.SINGLE_TOKEN_ATTENTION = self.previous

    def attention(self, **kwargs):
        return M.Attention(192, **kwargs).to(DEVICE).eval().requires_grad_(False)

    def test_single_token_forward_and_input_gradient(self):
        for seed in (11, 42):
            torch.manual_seed(seed)
            for batch in (1, 50, 200):
                for causal in (False, True):
                    for dropout in (0.0, 0.2):
                        with self.subTest(seed=seed, batch=batch, causal=causal, dropout=dropout):
                            module = self.attention(dropout=dropout)
                            # Strided input covers a non-contiguous caller tensor.
                            x = torch.randn(batch, 1, 384, device=DEVICE)[..., ::2]
                            cotangent = torch.randn_like(x)
                            call = lambda y: module(y, causal=causal)
                            ref = evaluate(call, (x,), cotangent, "off")
                            for mode in ("sdpa", "v"):
                                compare(evaluate(call, (x,), cotangent, mode), ref)
                            M.SINGLE_TOKEN_ATTENTION = "off"
                            with torch.inference_mode():
                                ref_inference = call(x)
                                for mode in ("sdpa", "v"):
                                    M.SINGLE_TOKEN_ATTENTION = mode
                                    compare((call(x),), (ref_inference,))

    def test_conditioning_gradient(self):
        # Do not leave AdaLN gates zero: that would hide attention gradient bugs.
        module = M.ConditionalBlock(192, 8, 64, 768).to(DEVICE).eval()
        torch.nn.init.normal_(module.adaLN_modulation[-1].weight, std=0.02)
        torch.nn.init.normal_(module.adaLN_modulation[-1].bias, std=0.02)
        module.requires_grad_(False)
        for batch in (1, 50, 200):
            x = torch.randn(batch, 1, 192, device=DEVICE)
            c = torch.randn_like(x)
            cotangent = torch.randn_like(x)
            ref = evaluate(module, (x, c), cotangent, "off")
            for mode in ("sdpa", "v"):
                compare(evaluate(module, (x, c), cotangent, mode), ref)

    def test_multi_token_and_training_fallback(self):
        for tokens, training in ((2, False), (6, False), (1, True)):
            for causal in (False, True):
                module = self.attention(dropout=0.2).train(training)
                x = torch.randn(3, tokens, 192, device=DEVICE)
                cotangent = torch.randn_like(x)
                call = lambda y: module(y, causal=causal)
                torch.manual_seed(99)
                ref = evaluate(call, (x,), cotangent, "off")
                for mode in ("sdpa", "v"):
                    torch.manual_seed(99)
                    with patch.object(M.F, "scaled_dot_product_attention",
                                      wraps=M.F.scaled_dot_product_attention) as sdpa:
                        compare(evaluate(call, (x,), cotangent, mode), ref, exact=True)
                        self.assertEqual(sdpa.call_count, 1)

    def test_identity_output_projection(self):
        module = self.attention(heads=1, dim_head=192)
        x = torch.randn(5, 1, 192, device=DEVICE)
        cotangent = torch.randn_like(x)
        ref = evaluate(module, (x,), cotangent, "off")
        for mode in ("sdpa", "v"):
            compare(evaluate(module, (x,), cotangent, mode), ref)

    def test_v_slice_and_unchanged_state(self):
        module = self.attention()
        original = {k: v.clone() for k, v in module.state_dict().items()}
        weight = module.to_qkv.weight
        x = torch.randn(50, 1, 192, device=DEVICE)
        for mode, expected_rows in (("sdpa", 1536), ("v", 512)):
            M.SINGLE_TOKEN_ATTENTION = mode
            seen = []
            linear = M.F.linear

            def record_linear(value, w, bias=None):
                seen.append(w)
                return linear(value, w, bias)

            with patch.object(M.F, "linear", side_effect=record_linear), patch.object(
                    M.F, "scaled_dot_product_attention", side_effect=AssertionError("unexpected SDPA")):
                module(x)
            w = seen[0]
            self.assertEqual(tuple(w.shape), (expected_rows, 192))
            self.assertEqual(w.stride(), (192, 1))
            self.assertTrue(w.is_contiguous())
            self.assertEqual(w.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr())
            self.assertEqual(w.storage_offset(), 1024 * 192 if mode == "v" else 0)
        current = module.state_dict()
        self.assertEqual(current.keys(), original.keys())
        for key in current:
            torch.testing.assert_close(current[key], original[key], atol=0, rtol=0)
        # Reloading different weights must update the slice, with no cached copy.
        replacement = {k: v + 0.01 for k, v in original.items()}
        module.load_state_dict(replacement, strict=True)
        cotangent = torch.randn_like(x)
        ref = evaluate(module, (x,), cotangent, "off")
        compare(evaluate(module, (x,), cotangent, "v"), ref)

    def test_cuda_graph_replay(self):
        if torch.device(DEVICE).type != "cuda":
            self.skipTest("CUDA graph replay requires a CUDA device")
        module = self.attention()
        for mode in ("sdpa", "v"):
            M.SINGLE_TOKEN_ATTENTION = mode
            sample = torch.randn(50, 1, 192, device=DEVICE, requires_grad=True)
            graph = torch.cuda.make_graphed_callables(lambda x: module(x), (sample,))
            for _ in range(3):
                x = torch.randn_like(sample)
                cotangent = torch.randn_like(x)
                ref = evaluate(module, (x,), cotangent, "off")
                compare(evaluate(graph, (x,), cotangent, mode), ref)


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    DEVICE = args.device
    if DEVICE == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AttentionChecks))
    report = {"passed": result.wasSuccessful(), "tests": result.testsRun,
              "skipped": [(str(test), reason) for test, reason in result.skipped], "max_abs": DIFFS,
              "torch": torch.__version__, "device": DEVICE,
              "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else None}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
