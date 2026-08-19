#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cilia_detector.dataset import iter_patches, make_training_arrays, save_patch_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract denoised channel-index-1 patches and ROI masks from analyzed TIFFs.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--glob", default="*.tif")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--channel-index", type=int, default=1, help="Zero-based image channel used as U-Net input.")
    parser.add_argument("--label-channel", type=int, default=2, help="ImageJ 1-based channel used for cilium ROIs.")
    parser.add_argument("--line-width", type=int, default=5)
    args = parser.parse_args()

    image_paths = sorted(args.image_dir.glob(args.glob))
    if not image_paths:
        raise SystemExit(f"No images matched {args.image_dir / args.glob}")

    total = 0
    for image_path in tqdm(image_paths, desc="images"):
        inputs, mask = make_training_arrays(
            image_path,
            label_channel=args.label_channel,
            channel_index=args.channel_index,
            line_width=args.line_width,
        )
        if mask.sum() == 0:
            tqdm.write(f"warning: no ROI mask extracted for {image_path.name}")
        patch_idx = 0
        for image_patch, mask_patch in iter_patches(inputs, mask, args.patch_size, args.stride):
            out_name = f"{image_path.stem}_patch{patch_idx:04d}.npz"
            save_patch_npz(args.out_dir / out_name, image_patch, mask_patch, str(image_path))
            patch_idx += 1
            total += 1
        tqdm.write(f"{image_path.name}: mask pixels={int(mask.sum())}, patches={patch_idx}")
    print(f"saved {total} patches to {args.out_dir}")


if __name__ == "__main__":
    main()
