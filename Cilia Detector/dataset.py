from __future__ import annotations

from pathlib import Path

import numpy as np

from .denoise import denoise_single_channel_input
from .io import extract_imagej_rois, read_stack, rois_to_mask


def make_training_arrays(
    image_path: str | Path,
    label_channel: int = 2,
    channel_index: int = 1,
    line_width: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    stack, metadata = read_stack(image_path)
    inputs = denoise_single_channel_input(stack, channel_index=channel_index)
    rois = extract_imagej_rois(metadata)
    mask = rois_to_mask(rois, stack.shape[-2:], channel=label_channel, line_width=line_width)
    if mask.sum() == 0:
        mask = rois_to_mask(rois, stack.shape[-2:], channel=None, line_width=line_width)
    return inputs, mask


def iter_patches(
    inputs: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    min_positive_pixels: int = 6,
):
    _, h, w = inputs.shape
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            image_patch = inputs[:, y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]
            if image_patch.shape[-2:] != (patch_size, patch_size):
                continue
            if mask_patch.sum() >= min_positive_pixels or np.random.random() < 0.08:
                yield image_patch.astype(np.float32), mask_patch.astype(np.float32)


def save_patch_npz(out_path: str | Path, image: np.ndarray, mask: np.ndarray, source: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, image=image, mask=mask, source=source)
