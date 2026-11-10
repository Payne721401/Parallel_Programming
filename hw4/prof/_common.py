"""Shared helpers for the HW4 profiling scripts in prof/. Not part of the submission.

Outputs go to $PROF_OUT (default ~/hw4-prof/manual), outside the hw4 directory, so the
judge never sees traces or captured tensors next to hw4.py.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def setup_paths() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def out_dir(*parts: str) -> Path:
    base = os.environ.get("PROF_OUT") or str(Path.home() / "hw4-prof" / "manual")
    path = Path(base).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def kernel_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("gemm", "gemv", "cutlass", "cublas", "xmma", "splitk")):
        return "gemm"
    if "layer_norm" in n or "layernorm" in n:
        return "layernorm"
    if "batch_norm" in n or "batchnorm" in n or "bn_fw" in n or "bn_bw" in n:
        return "batchnorm"
    if "softmax" in n:
        return "softmax"
    if "foreach" in n or "multi_tensor" in n:
        return "optimizer"
    if "reduce" in n:
        return "reduce"
    if "catarray" in n or ("cat" in n and "batched" in n):
        return "cat"
    if "index" in n or "gather" in n or "scatter" in n:
        return "index"
    if "copy" in n:
        return "copy"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "elementwise"
    return "other"


def short_name(name: str, width: int = 96) -> str:
    name = name.replace("void ", "").replace("at::native::", "").replace("(anonymous namespace)::", "")
    return name if len(name) <= width else name[: width - 3] + "..."


def analyze_trace(path, n_iters: int) -> dict:
    """Summarize a torch.profiler Chrome trace covering n_iters GD iterations.

    Device ops are attributed to a phase through their launch's correlation id and the
    innermost "prof:<phase>" record_function annotation that contains the launch.
    """
    with open(path) as stream:
        data = json.load(stream)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    kernels, memops, annotations = [], [], []
    launches = {}
    cpu_ops = 0
    launch_calls = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = str(event.get("cat", "")).lower()
        args = event.get("args") or {}
        if cat == "kernel":
            kernels.append(event)
        elif cat in ("gpu_memcpy", "gpu_memset"):
            memops.append(event)
        elif cat in ("cuda_runtime", "cuda_driver"):
            corr = args.get("correlation")
            if corr is not None:
                launches[corr] = event
            if "launchkernel" in str(event.get("name", "")).lower():
                launch_calls += 1
        elif cat == "user_annotation":
            name = str(event.get("name", ""))
            if name.startswith("prof:"):
                start = float(event.get("ts", 0))
                annotations.append((start, start + float(event.get("dur", 0)), name[5:]))
        elif cat == "cpu_op":
            cpu_ops += 1

    def phase_of(event):
        corr = (event.get("args") or {}).get("correlation")
        launch = launches.get(corr)
        if launch is None:
            return "unattributed"
        ts = float(launch.get("ts", 0))
        best = None
        for start, end, name in annotations:
            if start <= ts <= end and (best is None or end - start < best[0]):
                best = (end - start, name)
        return best[1] if best else "other"

    n = max(1, int(n_iters))
    cats = defaultdict(lambda: [0, 0.0])
    names = defaultdict(lambda: [0, 0.0])
    phases = defaultdict(lambda: [0, 0.0])
    busy_us = 0.0
    first, last = None, None
    for group, events_of in (("kernel", kernels), ("mem", memops)):
        for event in events_of:
            dur = float(event.get("dur", 0))
            ts = float(event.get("ts", 0))
            busy_us += dur
            first = ts if first is None else min(first, ts)
            last = ts + dur if last is None else max(last, ts + dur)
            name = str(event.get("name", ""))
            cat = kernel_category(name) if group == "kernel" else "memcpy/memset"
            cats[cat][0] += 1
            cats[cat][1] += dur
            names[name][0] += 1
            names[name][1] += dur
            phase = phase_of(event)
            phases[phase][0] += 1
            phases[phase][1] += dur

    def per_iter(table):
        return {
            key: {"count_per_iter": value[0] / n, "ms_per_iter": value[1] / 1000.0 / n}
            for key, value in sorted(table.items(), key=lambda item: -item[1][1])
        }

    top = sorted(names.items(), key=lambda item: -item[1][1])[:12]
    return {
        "iters": n,
        "kernels_per_iter": len(kernels) / n,
        "memops_per_iter": len(memops) / n,
        "launch_calls_per_iter": launch_calls / n,
        "cpu_ops_per_iter": cpu_ops / n,
        "gpu_busy_ms_per_iter": busy_us / 1000.0 / n,
        "gpu_span_ms_per_iter": ((last - first) / 1000.0 / n) if first is not None else None,
        "mean_kernel_us": busy_us / max(1, len(kernels) + len(memops)),
        "categories": per_iter(cats),
        "phases": per_iter(phases),
        "top_kernels": [
            {"name": short_name(name), "count_per_iter": value[0] / n, "ms_per_iter": value[1] / 1000.0 / n}
            for name, value in top
        ],
    }


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def table(headers, rows, digits: int = 2) -> str:
    cells = [[fmt(c, digits) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    for row in cells:
        lines.append("  ".join(
            cell.rjust(widths[i]) if i else cell.ljust(widths[i]) for i, cell in enumerate(row)
        ))
    return "\n".join(lines)
