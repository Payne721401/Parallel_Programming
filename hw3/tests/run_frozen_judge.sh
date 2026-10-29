#!/usr/bin/env bash
# Manual login-node handoff. The official judge creates its own Slurm job.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 tests/frozen_judge.py "$@"
