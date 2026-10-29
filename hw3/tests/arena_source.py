"""Add host-only arena ownership to a verified original-multistate snapshot.

This is a second, reversible transform. It accepts only the exact output of
lean_multistate_source.py (apart from CRLF normalization), changes no CUDA kernel
or update dispatch, and preserves the old allocation/free path when PP_ARENA=0.
The two storage headers must be frozen alongside the generated translation unit.
"""

from lean_multistate_source import restore_original_multistate


TRANSFORMER_VERSION = 1


_ALLOCATION_BEFORE = '''    check(cudaMalloc(&src, allocation_bytes), "allocate source");
    check(cudaMalloc(&dst, allocation_bytes), "allocate destination");
    if (inspect_sparse) sparse_work.allocate(p, profile.enabled);
'''

_ALLOCATION_AFTER = '''    if (storage.requested) {
        const size_t workspace_tiles = inspect_sparse
            ? size_t((p.width + TILE_W - 1) / TILE_W)
                * size_t((p.height + TILE_H - 1) / TILE_H) : 0;
        storage.allocate(allocation_bytes, inspect_sparse, workspace_tiles, profile.enabled);
        src = storage.pointer<uint8_t>(storage.layout.src);
        dst = storage.pointer<uint8_t>(storage.layout.dst);
        if (inspect_sparse) storage.bind_sparse(sparse_work, p, TILE_W, TILE_H);
    } else {
        check(cudaMalloc(&src, allocation_bytes), "allocate source");
        check(cudaMalloc(&dst, allocation_bytes), "allocate destination");
        if (inspect_sparse) sparse_work.allocate(p, profile.enabled);
        storage.allocation_calls = 2 + (inspect_sparse ? 5 + int(profile.enabled) : 0);
    }
'''

_RELEASE_BEFORE = '''    check(cudaFree(dst), "free destination");
    check(cudaFree(src), "free source");
    sparse_work.release();
'''

_RELEASE_AFTER = '''    if (storage.selected) {
        // Grid and workspace pointers are borrowed slices, including after swaps.
        // Graphs, diagnostics and final host output are finished before the owner.
        storage.release();
    } else {
        check(cudaFree(dst), "free destination");
        check(cudaFree(src), "free source");
        sparse_work.release();
        storage.release_calls = storage.allocation_calls;
    }
'''


PATCHES = (
    ("experiment scope comment",
     "// Only PP_MULTISTATE is supported here; the other new kernel families are absent.\n",
     "// Only PP_MULTISTATE and host-only PP_ARENA are supported here.\n"
     "// The other new kernel families are absent.\n"),
    ("host layout header", '#include "r1_multistate.hpp"\n',
     '#include "r1_multistate.hpp"\n#include "device_arena.hpp"\n'),
    ("host storage header", '#include "evolution_diag.cuh"\n',
     '#include "device_storage.cuh"\n#include "evolution_diag.cuh"\n'),
    ("storage construction", "    LeanMultistateExperiment multistate;\n",
     "    LeanMultistateExperiment multistate;\n    RunStorage storage;\n"),
    ("zero-step memory report", "        multistate.report(profile.enabled);\n",
     "        multistate.report(profile.enabled);\n        storage.report(profile.enabled);\n"),
    ("storage allocation", _ALLOCATION_BEFORE, _ALLOCATION_AFTER),
    ("storage release", _RELEASE_BEFORE, _RELEASE_AFTER),
    ("final memory report", "    multistate.report(profile.enabled);\n    if (profile.enabled)\n",
     "    multistate.report(profile.enabled);\n    storage.report(profile.enabled);\n    if (profile.enabled)\n"),
)


def _canonical(source):
    if not isinstance(source, bytes):
        raise TypeError("frozen source must be bytes")
    return source.replace(b"\r\n", b"\n")


def _replace_once(source, before, after, name):
    before, after = before.encode("utf-8"), after.encode("utf-8")
    count = source.count(before)
    if count != 1:
        raise ValueError(f"arena patch {name!r} requires one anchor; found {count}")
    return source.replace(before, after, 1)


def transform_multistate_arena(source):
    """Add arena host ownership to verified exact1352+dense-multistate bytes."""
    source = _canonical(source)
    restore_original_multistate(source)  # Reject unrelated or repeated edits.
    for name, before, after in PATCHES:
        source = _replace_once(source, before, after, name)
    return source


def restore_multistate_arena(source):
    """Remove arena edits and verify the entire restored multistate snapshot."""
    source = _canonical(source)
    for name, before, after in reversed(PATCHES):
        source = _replace_once(source, after, before, "inverse " + name)
    restore_original_multistate(source)
    return source


# Explicit aliases make the two-transform provenance clear at integration sites.
transform_original_multistate_arena = transform_multistate_arena
restore_original_multistate_arena = restore_multistate_arena
