#!/usr/bin/env python3
"""Read case descriptions only. Run locally, or inside Slurm on Nano5."""
import argparse
from pathlib import Path
import re


def describe(path):
    tokens = " ".join(line.split("#", 1)[0] for line in path.read_text(encoding="utf-8").splitlines()).split()
    width, height = map(int, tokens[:2])
    match = re.fullmatch(r"R(\d+),C(\d+),M([01]),S(\d+)\.\.(\d+),B(\d+)\.\.(\d+),N([MN])", tokens[2])
    if match is None:
        raise ValueError(f"invalid rule in {path}")
    radius, states, centre, slo, shi, blo, bhi = map(int, match.groups()[:7])
    shape = match.group(8)
    boundary, steps, count = tokens[3], int(tokens[4]), int(tokens[5])
    if len(tokens) != 6 + 6 * count:
        raise ValueError(f"invalid seed count in {path}")
    if not (1 <= radius <= 32 and 2 <= states <= 16 and
            2 * radius + 1 <= min(width, height) and max(width, height) <= 32768 and
            boundary in ("dead", "torus") and 0 <= steps <= 5000):
        raise ValueError(f"invalid parameters in {path}")
    neighbours = ((2*radius+1)**2-1 if shape == "M" else 2*radius*(radius+1)) + centre
    if not (centre <= slo <= shi <= neighbours and 1 <= blo <= bhi <= neighbours):
        raise ValueError(f"invalid rule bounds in {path}")
    area_sum = expected_sum = 0
    for i in range(count):
        x, y, w, h, percent, seed = map(int, tokens[6+6*i:12+6*i])
        if not (0 <= x <= width and 0 <= y <= height and 0 <= w <= width-x and
                0 <= h <= height-y and 0 <= percent <= 100 and 0 <= seed < 2**64):
            raise ValueError(f"invalid seed rectangle in {path}")
        area_sum += w*h
        expected_sum += w*h*percent/100
    cells = width*height
    return [str(path), width, height, radius, states, centre, shape, boundary, steps,
            neighbours, count, f"{area_sum/cells:.6f}", f"{expected_sum/cells:.6f}",
            cells*steps, cells*steps*neighbours]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    print("case\tW\tH\tR\tC\tM\tshape\tboundary\tT\tneighbours\trectangles\tarea_sum_ratio\texpected_sum_ratio\tcell_steps\tnaive_checks")
    for name in args.inputs:
        try:
            print("\t".join(map(str, describe(Path(name)))))
        except (OSError, ValueError, IndexError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
