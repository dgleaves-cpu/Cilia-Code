#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run U-Net prediction and object measurement for a folder of TIFFs.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--glob", default="*.tif")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    for image_path in sorted(args.image_dir.glob(args.glob)):
        mask_path = args.out_dir / f"{image_path.stem}_unet_mask.tif"
        objects_path = args.out_dir / f"{image_path.stem}_unet_objects.csv"
        predict_cmd = [
            sys.executable,
            str(root / "scripts" / "predict.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--image",
            str(image_path),
            "--out",
            str(mask_path),
        ]
        if args.threshold is not None:
            predict_cmd.extend(["--threshold", str(args.threshold)])
        subprocess.run(predict_cmd, check=True)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "classify_cilia.py"),
                "--image",
                str(image_path),
                "--mask",
                str(mask_path),
                "--out",
                str(objects_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

