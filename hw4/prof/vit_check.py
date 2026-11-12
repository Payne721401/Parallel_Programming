"""Compare native ViT to the installed transformers implementation.

--assets uses the released architecture and frozen ViT weights; otherwise test
the documented ViT-Tiny/14 shape plus small mask/pooling/interpolation variants.
This is a correctness probe; import and wall-time A/B must run in fresh processes.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys

import _common as C


def difference(torch, actual, expected):
    if actual.dtype != torch.float32 or actual.shape != expected.shape:
        raise AssertionError("wrong output dtype or shape")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError("non-finite output or gradient")
    return float((actual - expected).abs().max())


def compare_case(torch, hf, native, pixels, mask, backend, atol):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def context():
        return sdpa_kernel(SDPBackend.MATH) if backend == "math" else nullcontext()

    # Compare every encoder layer, not only the final CLS token.
    traces = ({}, {})
    handles = []
    for model, trace in zip((hf, native), traces):
        for name, layer in model.named_modules():
            if name == "embeddings" or name == "layernorm" or (
                    name.startswith("encoder.layer.") and name.count(".") == 2):
                def record(module, args, out, key=name, target=trace):
                    value = out[0] if isinstance(out, tuple) else out
                    target[key] = value.detach().clone()
                handles.append(layer.register_forward_hook(record))
    original = pixels.clone()
    try:
        with context(), torch.inference_mode():
            expected = hf(pixels, bool_masked_pos=mask, interpolate_pos_encoding=True)
            actual = native(pixels, bool_masked_pos=mask, interpolate_pos_encoding=True)
        forward = difference(torch, actual.last_hidden_state, expected.last_hidden_state)
        pool = 0.0 if expected.pooler_output is None else difference(torch, actual.pooler_output, expected.pooler_output)
        if traces[0].keys() != traces[1].keys():
            raise AssertionError("layer trace names differ")
        layers = {name: difference(torch, traces[1][name], traces[0][name]) for name in traces[0]}
    finally:
        for handle in handles:
            handle.remove()
    grad_values = []
    # A nontrivial fixed cotangent on the CLS vector exercises pixel gradients.
    cotangent = torch.randn(pixels.shape[0], hf.config.hidden_size, device=pixels.device)
    with context(), torch.enable_grad():
        for model in (hf, native):
            x = pixels.detach().clone().requires_grad_(True)
            output = model(x, bool_masked_pos=mask, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            grad_values.append(torch.autograd.grad(output, x, cotangent)[0].detach())
    gradient = difference(torch, grad_values[1], grad_values[0])
    torch.testing.assert_close(pixels, original, atol=0, rtol=0)
    maximum = max(forward, pool, gradient, *layers.values())
    return {"shape": list(pixels.shape), "masked": mask is not None, "backend": backend,
            "forward_max_abs": forward, "pool_max_abs": pool, "pixel_gradient_max_abs": gradient,
            "layer_max_abs": layers, "exact": maximum == 0, "passed": maximum <= atol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--assets", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    if not (0 <= args.atol < float("inf")):
        parser.error("atol must be finite and nonnegative")

    import torch
    import transformers
    from transformers import ViTConfig, ViTModel

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(12345)
    # Load only this source: no factory import can affect the reference model.
    spec = importlib.util.spec_from_file_location("hw4_native_vit", C.ROOT / "model/vit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    settings = {"vit_config": {"hidden_size": 192, "num_hidden_layers": 12,
                               "num_attention_heads": 3, "intermediate_size": 768,
                               "image_size": 224, "patch_size": 14, "layer_norm_eps": 1e-12},
                "attention_implementation": "sdpa", "use_mask_token": True, "add_pooling_layer": False}
    weights = None
    if args.assets:
        C.setup_paths()
        import runner

        all_weights, architecture, _ = runner.load_assets(args.device)
        settings = architecture["levels"][0]["encoder"]
        prefix = "jepas.0.encoder.encoder."
        weights = {key[len(prefix):]: value for key, value in all_weights.items() if key.startswith(prefix)}
        if not weights:
            raise RuntimeError("No released L1 ViT weights found")
    experiments = [("released" if args.assets else "tiny14", settings, weights)]
    # Small models exercise masks, interpolation, pooling, qkv_bias=False and eager.
    for implementation in ("sdpa", "eager"):
        small = {"vit_config": {"hidden_size": 24, "num_hidden_layers": 2,
                                "num_attention_heads": 3, "intermediate_size": 48,
                                "image_size": 28, "patch_size": 7, "qkv_bias": False,
                                "pooler_output_size": 12},
                 "attention_implementation": implementation, "use_mask_token": True,
                 "add_pooling_layer": True}
        experiments.append((f"small-{implementation}", small, None))
    report = {"passed": False, "torch": torch.__version__, "transformers": transformers.__version__,
              "device": args.device, "assets": args.assets, "atol": args.atol, "results": []}
    C.save_json(args.output, report)
    try:
        for label, settings, weights in experiments:
            config = ViTConfig.from_dict(settings["vit_config"])
            config._attn_implementation = settings["attention_implementation"]
            kwargs = {key: settings[key] for key in ("use_mask_token", "add_pooling_layer")}
            hf = ViTModel(config, **kwargs).to(args.device).eval().requires_grad_(False)
            if weights is not None:
                hf.load_state_dict(weights, strict=True)
            native = module.NativeViT(settings["vit_config"],
                                      attention_implementation=settings["attention_implementation"],
                                      **kwargs).to(args.device).eval().requires_grad_(False)
            native.load_state_dict(hf.state_dict(), strict=True)
            before = {key: value.clone() for key, value in hf.state_dict().items()}
            if dict(hf.named_buffers()).keys() != dict(native.named_buffers()).keys():
                raise AssertionError("ViT buffer names differ")
            size = config.image_size if isinstance(config.image_size, int) else config.image_size[0]
            patch = config.patch_size if isinstance(config.patch_size, int) else config.patch_size[0]
            variants = [(1, size, size, False), (2, size, size, False)]
            if label.startswith("small"):
                variants += [(1, size + patch, size, True)]
            for backend in ("default", "math"):
                for batch, height, width, masked in variants:
                    pixels = torch.randn(batch, config.num_channels, height, width, device=args.device)
                    mask = (torch.rand(batch, (height // patch) * (width // patch), device=args.device) > 0.5
                            if masked else None)
                    row = compare_case(torch, hf, native, pixels, mask, backend, args.atol)
                    row["model"] = label
                    report["results"].append(row)
                    print({key: value for key, value in row.items() if key != "layer_max_abs"}, flush=True)
                    C.save_json(args.output, report)
            for key, value in native.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
                torch.testing.assert_close(hf.state_dict()[key], before[key], atol=0, rtol=0)
        report["passed"] = bool(report["results"]) and all(row["passed"] for row in report["results"])
        report["exact"] = all(row["exact"] for row in report["results"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        C.save_json(args.output, report)
        print(f"ViT check passed={report['passed']}; {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
