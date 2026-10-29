"""Generate a submission candidate from exact first-stage CUDA prototype source.

Only host-side defaults change. Explicit 0/1 environment controls retain their
original meaning, and all GPU kernels, allocations, queues and transfers remain
the measured 3b88 implementation. This transforms a frozen copy, never the live
production source. The inverse verifies that no unrelated edit was introduced.
"""

import hashlib


BASE_REVISION = "3b88c9dccc89cb0060fbcb90991fa07a19ed7a33"
SOURCE_SHA256 = "b36f1630f343883256efd5cbab9794726940418cf9cf329258e864509b4ca0cf"
VERSION = 1


PATCHES = (
    ("automatic-default documentation",
     "// PP_MULTISTATE=1, PP_SPARSE_BITS=1 and PP_ROLLING=1 enable new opt-in kernels.\n",
     "// Submission defaults select measured kernels by rule and workload size.\n"
     "// PP_MULTISTATE, PP_SPARSE_BITS and PP_ROLLING accept explicit 0/1 overrides.\n"),
    ("automatic-default flag",
     "    static bool flag(const char *name) {\n"
     "        const char *value = std::getenv(name);\n"
     "        if (!value || std::strcmp(value, \"0\") == 0) return false;\n",
     "    static bool flag(const char *name, bool automatic) {\n"
     "        const char *value = std::getenv(name);\n"
     "        if (!value) return automatic;\n"
     "        if (std::strcmp(value, \"0\") == 0) return false;\n"),
    ("workload defaults constructor",
     "    GlobalExperiments() : multistate_requested(flag(\"PP_MULTISTATE\")),\n"
     "        rolling_requested(flag(\"PP_ROLLING\")), sparse_bits_requested(flag(\"PP_SPARSE_BITS\")) {}\n",
     "    explicit GlobalExperiments(const Problem &p)\n"
     "        : multistate_requested(flag(\"PP_MULTISTATE\",\n"
     "              p.rule.radius == 1 && p.rule.states >= 3 && p.rule.states <= 16\n"
     "              && size_t(p.width) * p.height >= (size_t(1) << 27) && p.steps >= 128)),\n"
     "          rolling_requested(flag(\"PP_ROLLING\",\n"
     "              p.rule.radius >= 2 && size_t(p.width) * p.height >= (size_t(1) << 20)\n"
     "              && p.steps >= 32)),\n"
     "          sparse_bits_requested(flag(\"PP_SPARSE_BITS\",\n"
     "              p.rule.radius == 1 && size_t(p.width) * p.height >= (size_t(1) << 28)\n"
     "              && p.steps >= 128)) {}\n"),
    ("problem-aware construction",
     "    GlobalExperiments global;\n",
     "    GlobalExperiments global(p);\n"),
)


def _canonical(source):
    if not isinstance(source, bytes):
        raise TypeError("frozen source must be bytes")
    return source.replace(b"\r\n", b"\n")


def _replace_once(source, before, after, name):
    before, after = before.encode("utf-8"), after.encode("utf-8")
    count = source.count(before)
    if count != 1:
        raise ValueError(f"release patch {name!r} requires one anchor; found {count}")
    return source.replace(before, after, 1)


def transform_release(source):
    """Apply general workload defaults to exact 3b88 source, returning LF bytes."""
    source = _canonical(source)
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("release requires exact 3b88 hw3-1.cu source")
    for name, before, after in PATCHES:
        source = _replace_once(source, before, after, name)
    return source


def restore_release(source):
    """Invert every named change and reject unrelated generated-source edits."""
    source = _canonical(source)
    for name, before, after in reversed(PATCHES):
        source = _replace_once(source, after, before, "inverse " + name)
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("generated release source has unrelated modifications")
    return source
