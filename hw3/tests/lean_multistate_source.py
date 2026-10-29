"""Build a reviewable dense-multistate-only experiment from exact 1352 source.

This transforms a frozen source copy; it never edits the production source tree.
Every edit is a named, unique insertion/replacement, and the inverse reconstructs
the original byte-for-byte. The existing multistate headers supply the only new
device kernels. Other experimental families are absent from this translation unit.
"""

import hashlib


ORIGINAL_BASELINE_REVISION = "1352bf87dad753b317915a17b436be2fce329804"
ORIGINAL_SOURCE_SHA256 = "e31e6839cc43fdbf99b3f385e82dbf30ba822802109de87b363486ada0ccdfb0"
TRANSFORMER_VERSION = 1


_EXPERIMENT = r'''// Frozen-source experiment: exact 1352 plus dense multistate only.
// Only PP_MULTISTATE is supported here; the other new kernel families are absent.
// No automatic policy or default promotion is introduced by this experiment.
struct LeanMultistateExperiment {
    bool requested = false, selected = false;
    LeanMultistateExperiment() {
        const char *value = std::getenv("PP_MULTISTATE");
        requested = value && std::strcmp(value, "1") == 0;
        if (value && !requested && std::strcmp(value, "0") != 0) {
            std::fprintf(stderr, "hw3-1: PP_MULTISTATE must be 0 or 1\n");
            std::exit(EXIT_FAILURE);
        }
    }
    void report(bool profile) const {
        if (profile) std::fprintf(stderr,
            "[PP_GLOBAL] multistate_requested=%d multistate_selected=%d "
            "rolling_requested=0 rolling_selected=0 sparse_bits_requested=0 sparse_bits_selected=0\n",
            int(requested), int(selected));
    }
};

'''


_DENSE_MULTISTATE = r'''    if (multistate.selected) {
        auto swap_buffers = [&]() { uint8_t *old = src; src = dst; dst = old; };
        check(r1_multistate_gpu::pack(src, reinterpret_cast<uint32_t *>(dst), p), "launch multistate pack");
        swap_buffers();
        for (int t = 0; t < iterations; ++t) {
            check(r1_multistate_gpu::step(reinterpret_cast<const uint32_t *>(src),
                                         reinterpret_cast<uint32_t *>(dst), p),
                  "launch dense multistate step");
            swap_buffers();
        }
        check(r1_multistate_gpu::unpack(reinterpret_cast<const uint32_t *>(src), dst, p),
              "launch multistate unpack");
        swap_buffers();
    } else if (bitplane.selected) {
'''


# All original lines outside these replacements are retained without rewriting.
# In particular, sparse workspace, initialization, H2D/D2H, graph/evolution,
# original update dispatch, profiling spans, and freeing retain their 1352 code.
PATCHES = (
    ("host header", '#include "r1_bits.hpp"\n',
     '#include "r1_bits.hpp"\n#include "r1_multistate.hpp"\n'),
    ("experiment metadata", "// Diagnostic only: no events, clock reads, or extra synchronization when off.\n",
     _EXPERIMENT + "// Diagnostic only: no events, clock reads, or extra synchronization when off.\n"),
    ("device header", '#include "r1_bitplane.cuh"\n',
     '#include "r1_bitplane.cuh"\n#include "r1_multistate.cuh"\n'),
    ("experiment construction", "    PhaseProfile profile;\n    EvolutionProfile evolution;\n",
     "    PhaseProfile profile;\n    LeanMultistateExperiment multistate;\n    EvolutionProfile evolution;\n"),
    ("zero-step dispatch report", '        profile.report("none", p); // all phase times zero; no CUDA events allocated\n',
     '        profile.report("none", p); // all phase times zero; no CUDA events allocated\n'
     '        multistate.report(profile.enabled);\n'),
    ("padded allocation", "    const size_t allocation_bytes = bit_eligible && bit_words*sizeof(uint32_t) > bytes\n"
     "                                  ? bit_words*sizeof(uint32_t) : bytes;\n",
     "    const bool multistate_eligible = multistate.requested && packed\n"
     "        && p.rule.states > 2 && p.rule.states <= 16 && !graph_requested && !evolution.enabled;\n"
     "    const size_t allocation_bytes = multistate_eligible ? r1_multistate::allocation_bytes(p)\n"
     "        : bit_eligible && bit_words*sizeof(uint32_t) > bytes\n"
     "                                  ? bit_words*sizeof(uint32_t) : bytes;\n"),
    ("dense selection", "    const bool sparse = inspect_sparse && sparse_work.initialize(src, dst, p, force_sparse);\n",
     "    const bool sparse = inspect_sparse && sparse_work.initialize(src, dst, p, force_sparse);\n"
     "    multistate.selected = multistate_eligible && !sparse;\n"),
    ("multistate chain", "    if (bitplane.selected) {\n", _DENSE_MULTISTATE),
    ("profile kernel", ': (bitplane.selected ? "dense_bits" : packed ? "dense_r1" : sliding ? "sliding" : tiled ? "tiled" : "naive"), p);\n',
     ': (multistate.selected ? "dense_multistate" : bitplane.selected ? "dense_bits" : packed ? "dense_r1" : sliding ? "sliding" : tiled ? "tiled" : "naive"), p);\n'),
    ("final dispatch report", "    bitplane.report(profile.enabled);\n    if (profile.enabled)\n",
     "    bitplane.report(profile.enabled);\n    multistate.report(profile.enabled);\n    if (profile.enabled)\n"),
    ("arch warp report", "int((arch & 2) && packed && !bitplane.selected), int(vn_diagonal_selected),\n",
     "int((arch & 2) && packed && !bitplane.selected && !multistate.selected), int(vn_diagonal_selected),\n"),
    ("arch shared report", "int((arch & 8) && packed && !bitplane.selected));\n",
     "int((arch & 8) && packed && !bitplane.selected && !multistate.selected));\n"),
)


def _canonical(source):
    if not isinstance(source, bytes):
        raise TypeError("frozen source must be bytes")
    return source.replace(b"\r\n", b"\n")


def _replace_once(source, before, after, name):
    before, after = before.encode("utf-8"), after.encode("utf-8")
    count = source.count(before)
    if count != 1:
        raise ValueError(f"lean multistate patch {name!r} requires one anchor; found {count}")
    return source.replace(before, after, 1)


def transform_original_multistate(source):
    """Return LF source with only dense multistate added to exact 1352 source."""
    source = _canonical(source)
    digest = hashlib.sha256(source).hexdigest()
    if digest != ORIGINAL_SOURCE_SHA256:
        raise ValueError("original-multistate requires exact 1352 hw3-1.cu source; SHA256=" + digest)
    for name, before, after in PATCHES:
        source = _replace_once(source, before, after, name)
    return source


def restore_original_multistate(source):
    """Invert the named edits and verify no other source content has changed."""
    source = _canonical(source)
    for name, before, after in reversed(PATCHES):
        source = _replace_once(source, after, before, "inverse " + name)
    if hashlib.sha256(source).hexdigest() != ORIGINAL_SOURCE_SHA256:
        raise ValueError("generated original-multistate source has unrelated modifications")
    return source
