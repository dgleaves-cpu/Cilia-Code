#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cilia_detector.classify import classify_objects
from cilia_detector.denoise import denoise_single_channel_input
from cilia_detector.io import extract_imagej_rois, read_stack, rois_to_mask, write_mask, write_stack
from cilia_detector.visualize import save_mask_overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export channel-2 ImageJ cilium ROIs as masks. This is useful for "
            "checking labels and creating reference masks before U-Net training."
        )
    )
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--glob", default="*.tif")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--channel-index", type=int, default=1)
    parser.add_argument("--label-channel", type=int, default=2)
    parser.add_argument("--line-width", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in sorted(args.image_dir.glob(args.glob)):
        stack, metadata = read_stack(image_path)
        rois = extract_imagej_rois(metadata)
        mask = rois_to_mask(rois, stack.shape[-2:], channel=args.label_channel, line_width=args.line_width)
        denoised = denoise_single_channel_input(stack, channel_index=args.channel_index)

        stem = image_path.stem
        mask_path = args.out_dir / f"{stem}_roi_cilia_mask.tif"
        denoised_path = args.out_dir / f"{stem}_denoised_channel2.tif"
        overlay_path = args.out_dir / f"{stem}_roi_overlay.png"
        object_path = args.out_dir / f"{stem}_roi_objects.csv"

        write_mask(mask_path, mask)
        write_stack(denoised_path, denoised)
        save_mask_overlay(stack, mask, str(overlay_path), channel=args.channel_index)
        objects = classify_objects(stack, mask)
        objects.to_csv(object_path, index=False)

        rows.append(
            {
                "image": image_path.name,
                "roi_count": len(rois),
                "mask_pixels": int(mask.sum()),
                "detected_objects": len(objects),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "objects_path": str(object_path),
            }
        )
        print(f"{image_path.name}: mask_pixels={int(mask.sum())}, objects={len(objects)}")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "roi_mask_summary.csv", index=False)
    print(f"saved summary to {args.out_dir / 'roi_mask_summary.csv'}")


if __name__ == "__main__":
    main()
