#!/usr/bin/env bash
# Run once after copying the assignment: bash scripts/setup_env.sh
# Builds .venv beside this package over a uv-managed Python 3.11, which brings
# its own headers and lives on a shared filesystem, so the venv runs on every
# GPU node and Triton finds Python.h. A system Python would tie the venv to
# the nodes that happen to have the same one installed.
set -Eeuo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${FOURROOM_ENV_DIR:-$PACKAGE_DIR/.venv}"
# On Nano5: the Python the judge's own environment is built over.
JUDGE_PYTHONS=/work/dogeon188/pp26/judge/python
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

fail() { printf 'setup_env: %s\n' "$*" >&2; exit 1; }

is_python311() {
    "$1" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' >/dev/null 2>&1
}

# The base interpreter: FOURROOM_BASE_PYTHON if set, else the judge's
# uv-managed Python 3.11, else one uv installs.
BASE_PYTHON=''
if [[ -n "${FOURROOM_BASE_PYTHON:-}" ]]; then
    BASE_PYTHON="$(command -v -- "$FOURROOM_BASE_PYTHON")" || \
        fail "Cannot execute FOURROOM_BASE_PYTHON: $FOURROOM_BASE_PYTHON"
elif [[ -d "$JUDGE_PYTHONS" ]]; then
    for candidate in $(printf '%s\n' "$JUDGE_PYTHONS"/cpython-3.11*/bin/python3.11 | sort -rV); do
        if is_python311 "$candidate"; then
            BASE_PYTHON="$candidate"
            break
        fi
    done
fi
if [[ -z "$BASE_PYTHON" ]] && command -v uv >/dev/null; then
    uv python install 3.11 >&2
    BASE_PYTHON="$(UV_PYTHON_PREFERENCE=only-managed uv python find 3.11)"
fi
[[ -n "$BASE_PYTHON" ]] || fail \
    "No uv-managed Python 3.11: $JUDGE_PYTHONS is missing and uv is not on PATH. Install uv, or set FOURROOM_BASE_PYTHON."
is_python311 "$BASE_PYTHON" || fail "$BASE_PYTHON is not Python 3.11."

# Rebuild a venv made over any other interpreter, such as one an earlier
# version of this script made over a system Python.
base_of() {
    "$1" -c 'import os, sys; print(os.path.realpath(sys._base_executable))' 2>/dev/null
}
if [[ "$(base_of "$ENV_DIR/bin/python" || true)" != "$(base_of "$BASE_PYTHON")" ]]; then
    printf 'setup_env: building %s over %s\n' "$ENV_DIR" "$BASE_PYTHON"
    "$BASE_PYTHON" -m venv --clear "$ENV_DIR"
fi
PYTHON="$ENV_DIR/bin/python"
[[ -x "$PYTHON" ]] || fail "Environment creation did not provide $PYTHON"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'

versions_match() {
    "$PYTHON" - "$PACKAGE_DIR/requirements.txt" <<'PY'
import importlib.metadata
from pathlib import Path
import sys
valid = True
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith(("#", "-")):
        continue
    name, expected = line.split("==", 1)
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        actual = "missing"
    print(f"{name}: {actual} (required {expected})")
    valid &= actual == expected
sys.exit(0 if valid else 1)
PY
}
if ! versions_match; then
    "$PYTHON" -m pip install --no-cache-dir --requirement "$PACKAGE_DIR/requirements.txt"
fi
versions_match
"$PYTHON" -m pip check
printf 'FourRoom environment: %s\n' "$ENV_DIR"
