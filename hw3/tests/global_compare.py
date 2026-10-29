#!/usr/bin/env python3
"""Frozen-source, two-binary correctness gates and same-allocation ABBA.

Build on the login node; compare only inside an existing Slurm allocation.
Unprofiled Compute samples and separate profile observations are never pooled.
"""
import argparse
import csv
import difflib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

from run_cases import PROFILE_TIMES, run_cases


BASELINE_REVISION = "1352bf8"
ORIGINAL_BASELINE_REVISION = "1352bf87dad753b317915a17b436be2fce329804"
FEATURES = {"disabled": (), "multistate": ("PP_MULTISTATE",), "rolling": ("PP_ROLLING",),
            "multistate-arena": ("PP_MULTISTATE", "PP_ARENA"),
            "sparse-bits": ("PP_SPARSE_BITS",),
            "combined": ("PP_MULTISTATE", "PP_ROLLING", "PP_SPARSE_BITS")}
OLD_FEATURES = FEATURES["combined"]
NEXT_FEATURES = ("PP_SPARSE_WARP", "PP_SPARSE_WIDE", "PP_SPARSE_OUTPUT")
FEATURES.update({"sparse-warp": (*OLD_FEATURES, NEXT_FEATURES[0]),
                 "sparse-wide": (*OLD_FEATURES, NEXT_FEATURES[1]),
                 "sparse-output": (*OLD_FEATURES, NEXT_FEATURES[2]),
                 "sparse-next": (*OLD_FEATURES, *NEXT_FEATURES)})
BASELINE_METHODS = ("disabled", "multistate", "rolling", "sparse-bits", "combined")
OUTPUT_REASONS = {"enabled", "disabled", "not_sparse", "pointer_attributes", "not_pinned_host",
                  "not_mapped", "invalid_sparse_metadata", "dense_fallback", "zero_steps"}
GROUPS = {"target": tuple(f"{i:02d}" for i in range(4, 13)),
          "large": ("04", "05", "06", "07"), "sparse": ("08", "09", "10"),
          "multistate": ("08", "10", "12"), "sparse-bits": ("08", "10")}
SOURCE_SUFFIXES = {".cu", ".cuh", ".hpp", ".h"}
CANDIDATE_VARIANTS = ("working", "original-multistate", "original-multistate-arena")
GATE_COUNTS = {"--quick": 22, "--multistate": 384, "--rolling": 128,
               "--sparse-bits": 136, "--sparse": 30, "--support": 28,
               "--sparse-warp": 32, "--sparse-wide": 200, "--sparse-pageable": 30,
               "--bitplane": 452, "--graphs": 38}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_fixed_source(left, right):
    # Git checkouts and Windows-to-Linux bundles may change only CRLF/LF.
    # Keep raw source hashes in the manifest; this guard compares fixed content.
    return left.read_bytes().replace(b"\r\n", b"\n") == right.read_bytes().replace(b"\r\n", b"\n")


def compile_flags(mode, host_compiler_exists):
    """Keep judge builds free of optional cluster compiler/runtime overrides."""
    if mode not in ("judge", "cluster"):
        raise ValueError("unknown compiler mode: " + str(mode))
    flags = ["-std=c++17", "-O3", "-arch=sm_90"]
    if mode == "cluster" and host_compiler_exists:
        flags += ["-ccbin", "/work/HPC_software/LMOD/gcc/12.5.0/bin/g++",
                  "-Xcompiler", "-static-libstdc++,-static-libgcc"]
    return flags


def write_table(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def number(value, positive=False):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or (positive and parsed == 0):
        raise ValueError("invalid duration: " + str(value))
    return parsed


def case_selection(explicit, group):
    selected = []
    if explicit is not None:
        for item in explicit.split(","):
            match = re.fullmatch(r"\s*(\d{1,2})(?:-(\d{1,2}))?\s*", item)
            if not match:
                raise ValueError("invalid --cases item: " + item)
            first, last = int(match[1]), int(match[2] or match[1])
            if not 1 <= first <= last <= 12:
                raise ValueError("case IDs must be in 01..12 and ranges ascending")
            selected.extend(f"{case:02d}" for case in range(first, last + 1))
    else:
        selected.extend(GROUPS[group])
    # A cheap unchanged-family guard is included even for a bounded subgroup.
    return tuple(dict.fromkeys([*selected, "11"]))


def configure(method, candidate, profiled=False, baseline_method="disabled", **overrides):
    for key in list(os.environ):
        if key.startswith("PP_"):
            del os.environ[key]
    os.environ.update(PP_ARCH="baseline", PP_GRAPH="0", PP_EVOLVE="0",
                      PP_PROFILE=str(int(profiled)), PP_SUPPORT="bbox",
                      PP_BITPLANE="auto", PP_BIT_LOGIC="compact", PP_BIT_ROWS="4",
                      PP_MULTISTATE="0", PP_ROLLING="0", PP_SPARSE_BITS="0",
                      PP_SPARSE_WARP="0", PP_SPARSE_WIDE="0", PP_SPARSE_OUTPUT="0", PP_ARENA="0")
    selected = method if candidate else baseline_method
    if selected != "disabled":
        os.environ.update({key: "1" for key in FEATURES[selected]})
    os.environ.update(overrides)
    return {key: os.environ[key] for key in sorted(os.environ) if key.startswith("PP_")}


def profile_fields(text):
    lines = [line for line in text.splitlines() if line.startswith("[PP_PROFILE] ")]
    if len(lines) != 1:
        raise ValueError("expected exactly one PP_PROFILE record")
    tokens = [token.split("=", 1) for token in lines[0].split()[1:]]
    fields = dict(tokens)
    if len(fields) != len(tokens):
        raise ValueError("duplicate profile field")
    if not fields.get("kernel"):
        raise ValueError("missing profile kernel")
    result = {"profile_status": "ok", "profile_kernel": fields["kernel"],
              **{key: number(fields[key]) for key in PROFILE_TIMES}}
    if "cells" in fields:
        result["profile_cells"] = unsigned_integer(fields["cells"], positive=True)
    return result


def unsigned_integer(value, positive=False):
    if not re.fullmatch(r"\d+", str(value)) or (positive and int(value) == 0):
        raise ValueError("invalid unsigned integer: " + str(value))
    return int(value)


def validate_gate(text, expected=None):
    matches = re.findall(r"^PASS (\d+)/(\d+)(?:\s|$)", text, re.M)
    if len(matches) != 1 or int(matches[0][0]) <= 0 or matches[0][0] != matches[0][1]:
        raise ValueError("missing or incomplete CPU reference gate")
    count = int(matches[0][0])
    if expected is not None and count != expected:
        raise ValueError(f"wrong CPU gate case count: {count}, expected {expected}")
    return count


def verify_global(text, configured, kernel):
    lines = [line for line in text.splitlines() if line.startswith("[PP_GLOBAL] ")]
    if len(lines) != 1:
        raise ValueError("candidate requires exactly one PP_GLOBAL dispatch record")
    tokens = [token.split("=", 1) for token in lines[0].split()[1:]]
    fields = dict(tokens)
    expected = {}
    for feature, selected in (("multistate", kernel == "dense_multistate"),
                              ("rolling", kernel in ("rolling", "sparse_rolling")),
                              ("sparse_bits", kernel == "sparse_bits")):
        requested = configured["PP_"+feature.upper()]
        expected[feature+"_requested"] = requested
        expected[feature+"_selected"] = str(int(selected))
        if selected and requested != "1":
            raise ValueError("unrequested global feature selected: " + feature)
    if fields != expected or len(tokens) != len(expected):
        raise ValueError("wrong global dispatch: " + str(fields) + "; expected " + str(expected))
    return {"global_"+key: value for key, value in fields.items()}


def verify_sparse_next(text, configured, kernel, cells=None):
    lines = [line for line in text.splitlines() if line.startswith("[PP_SPARSE_NEXT] ")]
    requested = {name: configured.get("PP_SPARSE_"+name.upper(), "0")
                 for name in ("warp", "wide", "output")}
    required = any(value == "1" for value in requested.values()) or kernel in ("sparse_warp", "sparse_wide")
    # Older binaries lack this record. Existing methods remain usable without
    # it; requesting a new feature always requires a complete dispatch record.
    if not lines and not required:
        return {"sparse_next_"+name+"_selected": "0" for name in requested}
    if len(lines) != 1:
        raise ValueError("expected exactly one PP_SPARSE_NEXT dispatch record")
    tokens = [token.split("=", 1) for token in lines[0].split()[1:]]
    fields = dict(tokens)
    expected_keys = {name+suffix for name in requested for suffix in ("_requested", "_selected")}
    expected_keys.update(("output_reason", "clear_bytes", "scatter_bytes", "tile_w", "tile_h"))
    if set(fields) != expected_keys or len(tokens) != len(expected_keys):
        raise ValueError("wrong sparse-next fields: " + str(fields))
    for name, value in requested.items():
        if fields[name+"_requested"] != value or fields[name+"_selected"] not in ("0", "1"):
            raise ValueError("wrong sparse-next requested/selected flags: " + name)
        if fields[name+"_selected"] == "1" and value != "1":
            raise ValueError("unrequested sparse-next feature selected: " + name)
    for name in ("warp", "wide"):
        if fields[name+"_selected"] != str(int(kernel == "sparse_"+name)):
            raise ValueError("sparse-next selection disagrees with profile kernel: " + name)
    selected = fields["output_selected"] == "1"
    reason = fields["output_reason"]
    if reason not in OUTPUT_REASONS or selected != (reason == "enabled"):
        raise ValueError("invalid sparse output reason/selection: " + reason)
    if requested["output"] == "0" and reason not in ("disabled", "zero_steps"):
        raise ValueError("disabled sparse output has an active fallback reason")
    if requested["output"] == "1" and reason == "disabled":
        raise ValueError("requested sparse output reported disabled")
    if selected and not kernel.startswith("sparse_"):
        raise ValueError("sparse output selected for a non-sparse kernel")
    if cells is None:
        raise ValueError("sparse-next dispatch requires a profile cell count")
    cells = unsigned_integer(cells, positive=True)
    for name in ("clear_bytes", "scatter_bytes"):
        value = unsigned_integer(fields[name])
        if value > cells or (not selected and value != 0):
            raise ValueError("invalid sparse output byte count: " + name)
    geometry = (unsigned_integer(fields["tile_w"], positive=True),
                unsigned_integer(fields["tile_h"], positive=True))
    if geometry != ((256, 16) if kernel == "sparse_wide" else (64, 32)):
        raise ValueError("sparse-next tile geometry disagrees with selected kernel")
    return {"sparse_next_"+key: value for key, value in fields.items()}


def verify_memory(text, configured, kernel):
    lines = [line for line in text.splitlines() if line.startswith("[PP_MEMORY] ")]
    requested = configured.get("PP_ARENA", "0")
    if not lines and requested == "0":
        return {"memory_arena_selected": "0"}
    if len(lines) != 1:
        raise ValueError("expected exactly one PP_MEMORY record")
    tokens = [token.split("=", 1) for token in lines[0].split()[1:]]
    fields = dict(tokens)
    keys = {"arena_requested", "arena_selected", "arena_bytes", "storage_allocations", "storage_frees"}
    if set(fields) != keys or len(tokens) != len(keys):
        raise ValueError("invalid memory ownership fields")
    selected = str(int(requested == "1" and kernel != "none"))
    if fields["arena_requested"] != requested or fields["arena_selected"] != selected:
        raise ValueError("memory arena dispatch disagrees with requested flags/kernel")
    arena_bytes = unsigned_integer(fields["arena_bytes"])
    allocations, releases = (unsigned_integer(fields[key]) for key in ("storage_allocations", "storage_frees"))
    if selected == "1":
        if not arena_bytes or arena_bytes % 256 or allocations != 1 or releases != 1:
            raise ValueError("arena must have one owner allocation and release with aligned capacity")
    elif arena_bytes or allocations != releases or allocations not in ((0,) if kernel == "none" else (2, 8)):
        raise ValueError("invalid legacy/zero-step storage ownership")
    return {"memory_"+key: value for key, value in fields.items()}


def verify_gate_dispatch(text, configured, count, feature=None, zero_feature=None):
    profiles = [line for line in text.splitlines() if line.startswith("[PP_PROFILE] ")]
    dispatch = [line for line in text.splitlines() if line.startswith("[PP_GLOBAL] ")]
    next_dispatch = [line for line in text.splitlines() if line.startswith("[PP_SPARSE_NEXT] ")]
    memory_dispatch = [line for line in text.splitlines() if line.startswith("[PP_MEMORY] ")]
    if len(profiles) != count or len(dispatch) != count:
        raise ValueError("CPU gate dispatch/profile record counts differ from case count")
    if next_dispatch and len(next_dispatch) != count:
        raise ValueError("CPU gate sparse-next record count differs from case count")
    if (memory_dispatch or configured.get("PP_ARENA") == "1") and len(memory_dispatch) != count:
        raise ValueError("CPU gate memory record count differs from case count")
    selected = {name+"_selected_cases": 0 for name in
                ("multistate", "rolling", "sparse_bits", "sparse_warp", "sparse_wide", "sparse_output", "arena")}
    for index, (profile, record) in enumerate(zip(profiles, dispatch)):
        parsed = profile_fields(profile)
        kernel = parsed["profile_kernel"]
        fields = verify_global(record, configured, kernel)
        for name in ("multistate_selected_cases", "rolling_selected_cases", "sparse_bits_selected_cases"):
            selected[name] += int(fields["global_"+name.replace("_cases", "")])
        fields = verify_sparse_next(next_dispatch[index] if next_dispatch else "", configured,
                                    kernel, parsed.get("profile_cells"))
        memory = verify_memory(memory_dispatch[index] if memory_dispatch else "", configured, kernel)
        selected["arena_selected_cases"] += int(memory["memory_arena_selected"])
        for name in ("warp", "wide", "output"):
            selected["sparse_"+name+"_selected_cases"] += int(fields["sparse_next_"+name+"_selected"])
    if feature and selected[feature+"_selected_cases"] == 0:
        raise ValueError("CPU gate never selected requested feature: " + feature)
    if zero_feature and selected[zero_feature+"_selected_cases"] != 0:
        raise ValueError("CPU gate selected a feature required to fall back: " + zero_feature)
    return selected


def gate_plan(method, baseline_method="disabled"):
    plan = []
    def add(option, role="candidate", required=None, zero=None, **overrides):
        plan.append(dict(option=option, role=role, required=required, zero=zero, overrides=overrides))
    add("--quick", "disabled")
    if baseline_method != "disabled":
        add("--quick", "baseline")
    add("--quick")
    if method in ("multistate", "multistate-arena", "combined"):
        add("--multistate", required="multistate", PP_ACTIVE="0")
    if method == "multistate-arena":
        add("--sparse", required="arena", PP_ACTIVE="1")
        add("--bitplane", required="arena", PP_ACTIVE="0", PP_BITPLANE="1")
        add("--graphs", required="arena", PP_ACTIVE="0", PP_GRAPH="1")
        add("--graphs", required="arena", PP_ACTIVE="1", PP_GRAPH="1")
    if method in ("rolling", "combined"):
        add("--rolling", required="rolling", PP_ACTIVE="0")
        add("--rolling", required="rolling", PP_ACTIVE="1")
    if method in ("sparse-bits", "combined"):
        add("--sparse-bits", required="sparse_bits", PP_ACTIVE="1")
        add("--sparse-bits", required="sparse_bits", PP_ACTIVE="1", PP_SUPPORT="tile")
    if method in ("sparse-warp", "sparse-next"):
        add("--sparse-warp", required="sparse_warp", PP_ACTIVE="1")
        add("--rolling", required="sparse_warp", PP_ACTIVE="1")
    if method in ("sparse-wide", "sparse-next"):
        add("--sparse-wide", required="sparse_wide", PP_ACTIVE="1")
        add("--sparse-wide", required="sparse_wide", PP_ACTIVE="1", PP_SUPPORT="tile")
    if method in ("sparse-output", "sparse-next"):
        add("--sparse", required="sparse_output", PP_ACTIVE="1")
        add("--sparse-pageable", zero="sparse_output", PP_ACTIVE="1")
    if method in ("rolling", "sparse-bits", "combined", "sparse-warp", "sparse-wide"):
        add("--sparse", PP_ACTIVE="1")
    if method in ("rolling", "sparse-bits", "combined", "sparse-warp", "sparse-wide", "sparse-output", "sparse-next"):
        add("--support", PP_ACTIVE="1")
    if method in ("rolling", "combined"):
        add("--sparse", PP_ACTIVE="1", PP_ARCH="reduce")
    return plan


def summarize(samples, cases, blocks):
    expected = 2 * blocks
    rows = []
    for case in cases:
        values = {}
        for method in ("baseline", "candidate"):
            group = [sample for sample in samples if sample["case_id"] == case
                     and sample["method"] == method and not int(sample["profiled"])]
            if len(group) != expected:
                raise ValueError(f"incomplete ABBA: {case} {method}: {len(group)}/{expected}")
            if any(sample["verdict"] not in ("AC", "TLE") for sample in group):
                raise ValueError("unverified output in timing comparison")
            values[method] = [number(sample["compute_s"], positive=True) for sample in group]
        baseline, candidate = values["baseline"], values["candidate"]
        before, after = statistics.mean(baseline), statistics.mean(candidate)
        rows.append(dict(case=case, samples_per_method=expected, baseline_mean_s=before,
                         candidate_mean_s=after, delta_candidate_minus_baseline_s=after-before,
                         baseline_median_s=statistics.median(baseline),
                         candidate_median_s=statistics.median(candidate),
                         baseline_min_s=min(baseline), baseline_max_s=max(baseline),
                         candidate_min_s=min(candidate), candidate_max_s=max(candidate),
                         baseline_over_candidate=before/after,
                         candidate_under_100ms_all_samples=int(max(candidate) < .1),
                         timing_status="unstable_samples" if
                         max(baseline)/min(baseline) > 1.2 or max(candidate)/min(candidate) > 1.2
                         else "range_within_20pct"))
    return rows


def phase_summary(samples, cases, blocks):
    rows = []
    for case in cases:
        for method in ("baseline", "candidate"):
            group = [sample for sample in samples if sample["case_id"] == case
                     and sample["method"] == method and int(sample["profiled"])]
            if len(group) != 2 * blocks or any(sample["profile_status"] != "ok" for sample in group):
                raise ValueError("incomplete separate profile observations")
            row = dict(case=case, method=method, observations=len(group),
                       kernels=";".join(sorted({sample["profile_kernel"] for sample in group})))
            for key in PROFILE_TIMES:
                values = [number(sample[key]) for sample in group]
                row.update({key+"_mean": statistics.mean(values), key+"_min": min(values),
                            key+"_max": max(values)})
            rows.append(row)
    return rows


def capture(argv, path, timeout=30, cwd=None):
    with path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    if result.returncode:
        raise ValueError("command failed; inspect " + str(path))


def git_bytes(*args):
    return subprocess.check_output(["git", *args], timeout=30,
                                   cwd=Path(__file__).resolve().parents[2])


def record_git(folder):
    for name, argv in (("revision.log", ["rev-parse", "HEAD"]),
                       ("status.log", ["status", "--short", "--untracked-files=all", "--", "hw3"]),
                       ("working-diff.log", ["diff", "HEAD", "--", "hw3"])):
        (folder/name).write_bytes(git_bytes(*argv))


def source_files(root):
    return sorted(path for path in root.rglob("*") if path.is_file()
                  and path.suffix in SOURCE_SUFFIXES)


def prepare_candidate_variant(variant, folder, baseline_source, candidate_source, baseline_revision):
    if variant not in CANDIDATE_VARIANTS:
        raise ValueError("unknown candidate variant: " + variant)
    if variant == "working":
        return dict(candidate_variant=variant), {}
    require_original_baseline(dict(baseline_revision=baseline_revision))
    from lean_multistate_source import transform_original_multistate, TRANSFORMER_VERSION
    original = (baseline_source/"hw3-1.cu").read_bytes()
    transformed = transform_original_multistate(original)
    for name in ("r1_multistate.hpp", "r1_multistate.cuh"):
        if not (candidate_source/name).is_file():
            raise ValueError("missing multistate header: " + name)
    # Keep every original production dependency, including r1_bits and architecture
    # kernels. Only the two multistate headers and current test gate are additions.
    for path in baseline_source.iterdir():
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            shutil.copyfile(path, candidate_source/path.name)
    (candidate_source/"hw3-1.cu").write_bytes(transformed)
    transformer = Path(__file__).with_name("lean_multistate_source.py")
    frozen_transformer = folder/"source-transformer.py"
    shutil.copyfile(transformer, frozen_transformer)
    source_diff = folder/"candidate-source.patch"
    source_diff.write_text("".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True), transformed.decode().splitlines(keepends=True),
        fromfile="1352bf8/hw3-1.cu", tofile="original-multistate/hw3-1.cu")), encoding="utf-8")
    metadata = dict(candidate_variant=variant, candidate_base_revision=baseline_revision,
                    source_transformer_version=TRANSFORMER_VERSION,
                    source_transformer_path=frozen_transformer.name,
                    source_transformer_sha256=sha256(frozen_transformer))
    hashes = {path.name: sha256(path) for path in (frozen_transformer, source_diff)}
    if variant == "original-multistate-arena":
        from arena_source import transform_multistate_arena, TRANSFORMER_VERSION as ARENA_VERSION
        for name in ("device_arena.hpp", "device_storage.cuh"):
            if not (candidate_source/name).is_file():
                raise ValueError("missing arena header: " + name)
        transformed = transform_multistate_arena(transformed)
        (candidate_source/"hw3-1.cu").write_bytes(transformed)
        arena_transformer = folder/"arena-source-transformer.py"
        shutil.copyfile(Path(__file__).with_name("arena_source.py"), arena_transformer)
        hashes[arena_transformer.name] = sha256(arena_transformer)
        source_diff.write_text("".join(difflib.unified_diff(
            original.decode().splitlines(keepends=True), transformed.decode().splitlines(keepends=True),
            fromfile="1352bf8/hw3-1.cu", tofile="original-multistate-arena/hw3-1.cu")), encoding="utf-8")
        hashes[source_diff.name] = sha256(source_diff)
        metadata.update(arena_source_transformer_version=ARENA_VERSION,
                        arena_source_transformer_path=arena_transformer.name,
                        arena_source_transformer_sha256=hashes[arena_transformer.name])
    return metadata, hashes


def prepare_build(args):
    folder = args.build_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    record_git(folder)
    baseline_revision = git_bytes("rev-parse", args.baseline_revision+"^{commit}").decode().strip()
    tree = git_bytes("archive", baseline_revision, "hw3")
    baseline_root = folder/"baseline-source"
    baseline_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(tree)) as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError("unsafe baseline archive member: " + member.name)
        archive.extractall(baseline_root)
    baseline_source = baseline_root/"hw3"
    candidate_source = folder/"candidate-source"/"hw3"
    candidate_source.mkdir(parents=True)
    current = Path.cwd()
    # Freeze working files, including new/untracked production headers. No binaries,
    # cases, tmp trees, unrelated directories, or ignored historical copies.
    inputs = list(current.glob("*"))
    for child in ("tests", "original_version"):
        if (current/child).is_dir():
            inputs.extend((current/child).rglob("*"))
    for path in inputs:
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            destination = candidate_source/path.relative_to(current)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    for fixed in ("main.cu", "ca.hpp"):
        if not same_fixed_source(candidate_source/fixed, baseline_source/fixed):
            raise ValueError("fixed file differs from baseline: " + fixed)
    variant = getattr(args, "candidate_variant", "working")
    variant_metadata, extra_hashes = prepare_candidate_variant(
        variant, folder, baseline_source, candidate_source, baseline_revision)
    print("Candidate variant: " + variant, flush=True)
    nvcc = args.nvcc or ("/work/HPC_software/LMOD/nvidia/packages/cuda-12.4/bin/nvcc"
                        if Path("/work/HPC_software/LMOD/nvidia/packages/cuda-12.4/bin/nvcc").is_file()
                        else "nvcc")
    compiler_mode = getattr(args, "compiler_mode", "judge")
    host_compiler = Path("/work/HPC_software/LMOD/gcc/12.5.0/bin/g++")
    flags = compile_flags(compiler_mode, host_compiler.is_file())
    print("Compiler mode: " + compiler_mode, flush=True)
    capture([nvcc, "--version"], folder/"compiler-version.log", timeout=15)
    manifest = dict(baseline_revision=baseline_revision,
                    candidate_revision=(folder/"revision.log").read_text().strip(),
                    source_hashes=extra_hashes, binaries={}, compile_commands=[],
                    compiler_mode=compiler_mode, **variant_metadata)
    for name, source in (("baseline", baseline_source), ("candidate", candidate_source)):
        for path in source_files(source):
            manifest["source_hashes"][str(path.relative_to(folder))] = sha256(path)
        binary = folder/(name+"-hw3-1")
        argv = [nvcc, *flags, "main.cu", "hw3-1.cu", "-o", str(binary)]
        manifest["compile_commands"].append(dict(cwd=str(source), argv=argv))
        print(f"Compiling {name}; output: {folder/(name+'-compile.log')}", flush=True)
        capture(argv, folder/(name+"-compile.log"), timeout=300, cwd=source)
        manifest["binaries"][name] = dict(path=binary.name, sha256=sha256(binary))
    gate = folder/"candidate-test_hw3-1"
    argv = [nvcc, *flags, "-I.", "tests/test_hw3-1.cu", "hw3-1.cu", "-o", str(gate)]
    manifest["compile_commands"].append(dict(cwd=str(candidate_source), argv=argv))
    print(f"Compiling candidate reference gate; output: {folder/'gate-compile.log'}", flush=True)
    capture(argv, folder/"gate-compile.log", timeout=300, cwd=candidate_source)
    manifest["binaries"]["gate"] = dict(path=gate.name, sha256=sha256(gate))
    for relative, digest in manifest["source_hashes"].items():
        if sha256(folder/relative) != digest:
            raise ValueError("source changed while compiling: " + relative)
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    (folder/"BUILT").write_text("Both source snapshots and candidate reference gate compiled.\n")
    print("Frozen build: " + str(folder), flush=True)
    print("No GPU was allocated or used. Start compare through srun/sbatch.", flush=True)


def verify_manifest(folder):
    if not (folder/"BUILT").is_file():
        raise ValueError("incomplete build; missing BUILT")
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    binaries = {}
    for name, item in manifest["binaries"].items():
        path = folder/item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("binary changed since frozen build: " + str(path))
        binaries[name] = path.resolve()
    for relative, digest in manifest["source_hashes"].items():
        if sha256(folder/relative) != digest:
            raise ValueError("source snapshot changed since build: " + relative)
    return manifest, binaries


def safe_environment():
    names = ("CUDA_MODULE_LOADING", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS",
             "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS",
             "SLURM_TIME_LIMIT", "SLURM_JOB_PARTITION", "LD_LIBRARY_PATH")
    return {key: os.environ.get(key, "unset") for key in names}


def comparison_binaries(manifest, binaries, baseline_binary="baseline"):
    """Select comparison roles without changing the frozen build or its hashes."""
    if baseline_binary not in ("baseline", "candidate"):
        raise ValueError("baseline binary must be baseline or candidate")
    selected = dict(binaries)
    selected["baseline"] = binaries[baseline_binary]
    provenance = {}
    for role, frozen_name in (("baseline", baseline_binary), ("candidate", "candidate")):
        provenance[role] = dict(frozen_binary=frozen_name,
                                source_revision=manifest[frozen_name+"_revision"],
                                source_snapshot=frozen_name+"-source/hw3",
                                source_variant=(manifest.get("candidate_variant", "working")
                                                if frozen_name == "candidate" else "original"),
                                base_revision=(manifest.get("candidate_base_revision", manifest["candidate_revision"])
                                               if frozen_name == "candidate" else manifest["baseline_revision"]),
                                binary_path=str(binaries[frozen_name]),
                                binary_sha256=manifest["binaries"][frozen_name]["sha256"])
    return selected, provenance


def require_original_baseline(manifest):
    actual = manifest.get("baseline_revision", "missing")
    if actual != ORIGINAL_BASELINE_REVISION:
        raise ValueError("regression control requires frozen baseline " + ORIGINAL_BASELINE_REVISION
                         + "; build contains " + str(actual) + ". Create a new build with --baseline-revision 1352bf8.")


def verify_variant_methods(manifest, method, baseline_method, baseline_binary="baseline"):
    variant = manifest.get("candidate_variant", "working")
    if variant == "original-multistate-arena":
        if method not in ("disabled", "multistate", "multistate-arena") or baseline_method not in ("disabled", "multistate"):
            raise ValueError("original-multistate-arena only supports multistate/arena controls")
        if baseline_method == "multistate" and baseline_binary != "candidate":
            raise ValueError("arena control requires --baseline-binary candidate for a multistate baseline")
        if method == "multistate-arena" and baseline_method != "multistate":
            raise ValueError("arena control requires --baseline-method multistate to keep generation kernels identical")
        return
    if variant != "original-multistate":
        return
    if method not in ("disabled", "multistate") or baseline_method != "disabled":
        raise ValueError("original-multistate control supports only disabled/multistate with a disabled baseline")


def compare(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("compare requires an existing srun/sbatch allocation")
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 540:
        raise ValueError("--budget-seconds must be positive and at most 540 (9 minutes)")
    if not math.isfinite(args.wall_timeout) or not 0 < args.wall_timeout <= 30:
        raise ValueError("--wall-timeout must be positive and at most 30")
    if not shutil.which(args.checker):
        raise ValueError("official checker missing: " + args.checker)
    baseline_method = getattr(args, "baseline_method", "disabled")
    deadline = time.monotonic()+args.budget_seconds
    folder = args.outdir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder/"started").write_text("single-allocation, frozen two-binary comparison\n")
    manifest, frozen_binaries = verify_manifest(args.build_dir.resolve())
    baseline_binary = getattr(args, "baseline_binary", "baseline")
    verify_variant_methods(manifest, args.method, baseline_method, baseline_binary)
    binaries, provenance = comparison_binaries(manifest, frozen_binaries, baseline_binary)
    gate_provenance = dict(frozen_binary="gate", source_revision=manifest["candidate_revision"],
                           source_variant=manifest.get("candidate_variant", "working"),
                           base_revision=manifest.get("candidate_base_revision", manifest["candidate_revision"]),
                           source_snapshot="candidate-source/hw3", binary_path=str(frozen_binaries["gate"]),
                           binary_sha256=manifest["binaries"]["gate"]["sha256"])
    shutil.copyfile(args.build_dir/"manifest.json", folder/"build-manifest.json")
    for name in ("revision.log", "status.log", "working-diff.log"):
        shutil.copyfile(args.build_dir/name, folder/("build-"+name))
    shutil.copyfile(args.build_dir/"compiler-version.log", folder/"build-compiler-version.log")
    record_git(folder)
    (folder/"environment.json").write_text(json.dumps(safe_environment(), indent=2)+"\n")
    capture(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pci.bus_id", "--format=csv"],
            folder/"device.log", timeout=15)
    capture(["nvidia-smi", "-q", "-d", "PERFORMANCE,CLOCK,POWER"], folder/"device-state-start.log", timeout=15)
    cases = case_selection(args.cases, args.group)
    case_rows = []
    for case in cases:
        path = Path("cases")/(case+".txt")
        golden = path.with_suffix(".golden.txt")
        if not golden.is_file():
            golden = path.with_suffix(".out")
        header = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
        header = [line for line in header if line]
        if len(header) < 4 or not golden.is_file():
            raise ValueError("missing input header or golden for case " + case)
        case_rows.append(dict(case=case, dimensions=header[0], rule=header[1], boundary=header[2],
                              steps=header[3], input_sha256=sha256(path), golden_sha256=sha256(golden)))
    write_table(folder/"case-inputs.tsv", case_rows)
    paths = [Path(__file__).resolve(), Path(__file__).with_name("run_global_compare.sh"),
             Path(__file__).with_name("run_cases.py"), *frozen_binaries.values()]
    (folder/"sha256.log").write_text("\n".join(sha256(path)+"  "+str(path) for path in paths)+"\n")
    configuration = dict(method=args.method, baseline_method=baseline_method, cases=cases, unprofiled_blocks=args.blocks,
                         separate_profile_blocks=args.profile_blocks, guard_case="11",
                         budget_seconds=args.budget_seconds, baseline_binary=baseline_binary,
                         baseline_revision=provenance["baseline"]["source_revision"],
                         frozen_baseline_revision=manifest["baseline_revision"], binary_provenance=provenance,
                         gate_binary_provenance=gate_provenance)
    (folder/"configuration.json").write_text(json.dumps(configuration, indent=2)+"\n")
    print("Baseline: " + provenance["baseline"]["source_revision"]
          + " (frozen " + baseline_binary + " binary)", flush=True)
    print("Baseline flags: " + baseline_method + "; candidate flags: " + args.method, flush=True)
    print("Candidate build HEAD: " + manifest["candidate_revision"], flush=True)
    print("Candidate source variant: " + manifest.get("candidate_variant", "working")
          + "; base: " + manifest.get("candidate_base_revision", manifest["candidate_revision"]), flush=True)
    for role in ("baseline", "candidate"):
        print(f"{role.title()} binary SHA256: {provenance[role]['binary_sha256']}", flush=True)
    print("Reference gates use the frozen candidate-source test binary.", flush=True)
    print("Cases: " + ",".join(cases) + "; 11 is the unchanged-family guard", flush=True)
    def remaining(required=0):
        value = deadline-time.monotonic()
        if value <= required:
            raise ValueError("comparison budget exhausted; partial logs retained, no COMPLETE")
        return value
    environments = []
    gates = []
    for index, spec in enumerate(gate_plan(args.method, baseline_method)):
        enabled, option = spec["role"] == "candidate", spec["option"]
        gate_baseline = baseline_method if spec["role"] == "baseline" else "disabled"
        configured = configure(args.method, enabled, True, baseline_method=gate_baseline, **spec["overrides"])
        environments.append(dict(kind="gate", index=index, option=option,
                                 method=spec["role"], env=configured, binary_provenance=gate_provenance))
        (folder/"effective-environments.json").write_text(json.dumps(environments, indent=2)+"\n")
        log = folder/f"gate-{index:02d}-{option[2:]}.log"
        capture([str(binaries["gate"]), option], log, timeout=min(120, remaining(1)))
        text = log.read_text(encoding="utf-8")
        count = validate_gate(text, GATE_COUNTS[option])
        selected = verify_gate_dispatch(text, configured, count, spec["required"], spec["zero"])
        gates.append(dict(gate=index, feature_enabled=int(enabled), role=spec["role"], option=option, passed=count,
                          active=configured.get("PP_ACTIVE", "auto"), support=configured["PP_SUPPORT"],
                          arch=configured["PP_ARCH"], **selected, log=log.name))
        write_table(folder/"gates.tsv", gates)
        print(f"Gate {index} {option}: PASS {count}/{count}; enabled={enabled}", flush=True)
    samples = []
    round_number = 0
    for profiled, blocks in ((False, args.blocks), (True, args.profile_blocks)):
        for block in range(1, blocks+1):
            for slot, method in enumerate(("baseline", "candidate", "candidate", "baseline"), 1):
                round_number += 1
                configured = configure(args.method, method == "candidate", profiled,
                                       baseline_method=baseline_method)
                environments.append(dict(kind="round", index=round_number, method=method, env=configured,
                                         binary_provenance=provenance[method]))
                (folder/"effective-environments.json").write_text(json.dumps(environments, indent=2)+"\n")
                order = cases if round_number % 2 else tuple(reversed(cases))
                round_folder = folder/f"round-{round_number:02d}-{method}-profile{int(profiled)}"
                round_folder.mkdir()
                print(f"Round {round_number}: {method}, PP_PROFILE={int(profiled)}, block={block}, slot={slot}", flush=True)
                for position, case in enumerate(order):
                    remaining(args.wall_timeout+1)
                    case_folder = round_folder/case
                    status = run_cases(argparse.Namespace(inputs=[f"cases/{case}.txt"], binary=str(binaries[method]),
                                       checker=args.checker, wall_timeout=args.wall_timeout,
                                       limit=None, outdir=str(case_folder)))
                    with (case_folder/"results.tsv").open(encoding="utf-8", newline="") as stream:
                        results = list(csv.DictReader(stream, delimiter="\t"))
                    if len(results) != 1:
                        raise ValueError("incomplete case result: " + str(case_folder))
                    row = results[0]
                    raw = (case_folder/f"00-{case}.log").read_text(encoding="utf-8", errors="replace")
                    if profiled:
                        row.update(profile_fields(raw))
                        if method == "candidate" or baseline_method != "disabled" or baseline_binary == "candidate":
                            row.update(verify_global(raw, configured, row["profile_kernel"]))
                            row.update(verify_sparse_next(raw, configured, row["profile_kernel"],
                                                          row.get("profile_cells")))
                            row.update(verify_memory(raw, configured, row["profile_kernel"]))
                            memory_record = next((line for line in raw.splitlines() if line.startswith("[PP_MEMORY] ")), None)
                            if memory_record:
                                print(f"  Case {case}: {memory_record}", flush=True)
                            if method == "candidate" and any(configured[key] == "1" for key in NEXT_FEATURES):
                                dispatch = next(line for line in raw.splitlines()
                                                if line.startswith("[PP_SPARSE_NEXT] "))
                                print(f"  Case {case}: {dispatch}", flush=True)
                    elif row["profile_status"] != "absent" or "[PP_MEMORY]" in raw:
                        raise ValueError("instrumentation in unprofiled timing round")
                    sample = dict(round=round_number, block=block, slot=slot, position=position,
                                  method=method, profiled=int(profiled), case_id=case,
                                  frozen_binary=provenance[method]["frozen_binary"],
                                  source_revision=provenance[method]["source_revision"],
                                  binary_sha256=provenance[method]["binary_sha256"],
                                  raw_log=str((case_folder/f"00-{case}.log").relative_to(folder)), **row)
                    samples.append(sample)
                    write_table(folder/"samples.tsv", samples)
                    # Correct TLEs remain measurements. WA/RE/timeout remains in the
                    # raw table, but stops this experiment instead of entering ratios.
                    if status not in (0, 1) or row["verdict"] not in ("AC", "TLE"):
                        raise ValueError("unverified public output: " + str(case_folder))
                    number(row["compute_s"], positive=True)
    summary = summarize(samples, cases, args.blocks)
    write_table(folder/"summary.tsv", summary)
    write_table(folder/"phases.tsv", phase_summary(samples, cases, args.profile_blocks))
    verify_manifest(args.build_dir.resolve())
    for row in summary:
        print("{case}: {baseline_mean_s:.6f} -> {candidate_mean_s:.6f}s; "
              "A/B={baseline_over_candidate:.3f}x; {timing_status}".format(**row), flush=True)
    total_before = sum(row["baseline_mean_s"] for row in summary)
    total_after = sum(row["candidate_mean_s"] for row in summary)
    print(f"Selected-case mean totals (includes 11 guard): {total_before:.6f} -> {total_after:.6f}s", flush=True)
    print("All samples retained; profile observations excluded from speedup and totals. No default promotion.", flush=True)
    (folder/"COMPLETE").write_text("All requested gates, unprofiled ABBA, separate profiles, and checksum checks completed.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="compile frozen baseline/candidate; no GPU work")
    build.add_argument("--build-dir", type=Path, required=True)
    build.add_argument("--baseline-revision", default=BASELINE_REVISION)
    build.add_argument("--candidate-variant", choices=CANDIDATE_VARIANTS, default="working",
                       help="original-multistate generates exact1352 plus dense multistate; keeps working production untouched")
    build.add_argument("--nvcc")
    build.add_argument("--compiler-mode", choices=("judge", "cluster"), default="judge",
                       help="judge uses the official flags; cluster opts into GCC/static-runtime overrides")
    run = subparsers.add_parser("compare", help="run only inside existing Slurm allocation")
    run.add_argument("--build-dir", type=Path, required=True)
    run.add_argument("--outdir", type=Path, required=True)
    run.add_argument("--method", choices=FEATURES, default="combined")
    run.add_argument("--baseline-method", choices=BASELINE_METHODS, default="disabled",
                     help="feature flags for the baseline role; defaults to all disabled")
    run.add_argument("--baseline-binary", choices=("baseline", "candidate"), default="baseline",
                     help="use candidate to isolate flags in the exact same frozen binary")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--group", choices=GROUPS, default="target")
    selection.add_argument("--cases", help="comma-separated case IDs/ranges; 11 guard always included")
    run.add_argument("--blocks", type=int, choices=range(1, 4), default=1)
    run.add_argument("--profile-blocks", type=int, choices=range(1, 3), default=1)
    run.add_argument("--wall-timeout", type=float, default=20)
    run.add_argument("--budget-seconds", type=float, default=420)
    run.add_argument("--checker", default="hw3-1-check")
    args = parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parent.parent)
    if args.command == "build":
        prepare_build(args)
    else:
        compare(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as error:
        print("global comparison failed: " + str(error), file=sys.stderr)
        sys.exit(1)
