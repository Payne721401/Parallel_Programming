#!/usr/bin/env python3
"""Permit correct-but-slow A/B runs; never suppress WA/RE/timeout or runner errors."""
import csv
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 3:
        raise ValueError("usage: check_timing_results.py results.tsv runner_exit_code")
    status = int(sys.argv[2])
    with Path(sys.argv[1]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if status not in (0, 1) or not rows or any(row.get("verdict") not in ("AC", "TLE") for row in rows):
        print("A/B stopped: an output was not verified correct; inspect results.tsv", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
