#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cilia_detector.denoise import denoise_single_channel_input
from cilia_detector.io import read_stack, write_stack


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise one selected image channel and save a one-channel float TIFF.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--channel-index", type=int, default=1)
    args = parser.parse_args()
    stack, _ = read_stack(args.image)
    denoised = denoise_single_channel_input(stack, channel_index=args.channel_index)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_stack(args.out, denoised)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
