#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tifffile as tiff

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cilia_detector.classify import classify_objects
from cilia_detector.io import read_stack


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify and measure predicted cilium objects.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--min-length-px", type=float, default=5.0)
    args = parser.parse_args()

    stack, _ = read_stack(args.image)
    mask = tiff.imread(args.mask)
    table = classify_objects(stack, mask, min_area=args.min_area, min_length_px=args.min_length_px)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"saved {len(table)} objects to {args.out}")


if __name__ == "__main__":
    main()
