#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from skimage import morphology

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cilia_detector.channel2_pipeline import CHANNEL_INDEX, load_unet_model
from cilia_detector.denoise import denoise_single_channel_input
from cilia_detector.io import read_stack, write_mask
from cilia_detector.model import UNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict cilium mask from channel-index 1 using a trained U-Net.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--channel-index", type=int, default=CHANNEL_INDEX, help="Must be 1: the zero-based second image channel.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.channel_index != CHANNEL_INDEX:
        raise SystemExit(
            f"This project only supports --channel-index {CHANNEL_INDEX}, the zero-based second image channel."
        )
    try:
        model, checkpoint = load_unet_model(args.checkpoint, device=device)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"Could not load a compatible Channel-2 U-Net checkpoint: {exc}") from exc

    stack, _ = read_stack(args.image)
    inputs = denoise_single_channel_input(stack, channel_index=CHANNEL_INDEX)
    prob = predict_tiled(model, inputs, device, args.tile_size, args.overlap)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)
    mask = prob >= threshold
    mask = morphology.remove_small_objects(mask, min_size=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_mask(args.out, mask.astype(np.uint8))
    print(f"saved {args.out}")


def predict_tiled(model: UNet, inputs: np.ndarray, device: torch.device, tile_size: int, overlap: int) -> np.ndarray:
    _, h, w = inputs.shape
    stride = tile_size - overlap
    prob = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y0 = min(y, max(0, h - tile_size))
                x0 = min(x, max(0, w - tile_size))
                tile = inputs[:, y0 : y0 + tile_size, x0 : x0 + tile_size]
                tensor = torch.from_numpy(tile[None, ...]).float().to(device)
                pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                prob[y0 : y0 + tile_size, x0 : x0 + tile_size] += pred
                weight[y0 : y0 + tile_size, x0 : x0 + tile_size] += 1
    return prob / np.maximum(weight, 1e-6)


if __name__ == "__main__":
    main()
