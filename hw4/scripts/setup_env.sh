#!/usr/bin/env bash
# Run once after copying the assignment: bash scripts/setup_env.sh
# Defaults to an isolated Python 3.11 environment beside this package.
set -Eeuo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${FOURROOM_ENV_DIR:-$PACKAGE_DIR/.venv}"
SHARE_PACKAGES="${FOURROOM_SYSTEM_SITE_PACKAGES:-0}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

fail() { printf 'setup_env: %s\n' "$*" >&2; exit 1; }

# Accept an executable path or an executable available on PATH.
find_command() {
    if [[ "$1" == */* ]]; then
        [[ -x "$1" && ! -d "$1" ]] || return 1
        printf '%s\n' "$1"
    else
        command -v -- "$1"
    fi
}

is_python311() {
    "$1" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' >/dev/null 2>&1
}

[[ "$SHARE_PACKAGES" == 0 || "$SHARE_PACKAGES" == 1 ]] || \
    fail 'FOURROOM_SYSTEM_SITE_PACKAGES must be 0 or 1.'

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    BASE_PYTHON=''
    if [[ -n "${FOURROOM_BASE_PYTHON:-}" ]]; then
        BASE_PYTHON="$(find_command "$FOURROOM_BASE_PYTHON")" || \
            fail "Cannot execute FOURROOM_BASE_PYTHON: $FOURROOM_BASE_PYTHON"
        is_python311 "$BASE_PYTHON" || fail 'FOURROOM_BASE_PYTHON must be Python 3.11.'
    else
        for candidate in python3.11 python3 python; do
            candidate_path="$(find_command "$candidate")" || continue
            if is_python311 "$candidate_path"; then
                BASE_PYTHON="$candidate_path"
                break
            fi
        done
    fi

    if [[ -n "$BASE_PYTHON" ]]; then
        venv_options=()
        if [[ "$SHARE_PACKAGES" == 1 ]]; then
            # Explicit opt-in only; pip installs still target the new environment.
            venv_options+=(--system-site-packages)
        fi
        "$BASE_PYTHON" -m venv "${venv_options[@]}" "$ENV_DIR"
    else
        [[ "$SHARE_PACKAGES" == 0 ]] || \
            fail 'Package sharing needs an existing Python 3.11; set FOURROOM_BASE_PYTHON.'
        CONDA_BIN=''
        if [[ -n "${FOURROOM_CONDA_EXE:-}" ]]; then
            CONDA_BIN="$(find_command "$FOURROOM_CONDA_EXE")" || \
                fail "Cannot execute FOURROOM_CONDA_EXE: $FOURROOM_CONDA_EXE"
        else
            for candidate in "${CONDA_EXE:-}" conda \
                "${HOME:?}/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
                [[ -n "$candidate" ]] || continue
                CONDA_BIN="$(find_command "$candidate")" && break
            done
        fi
        [[ -n "$CONDA_BIN" ]] || fail \
            'Python 3.11 and Conda were not found. Install Miniconda, or set FOURROOM_BASE_PYTHON / FOURROOM_CONDA_EXE.'
        "$CONDA_BIN" create --yes --prefix "$ENV_DIR" \
            --override-channels --channel conda-forge python=3.11 pip
    fi
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
