from __future__ import annotations

import numpy as np
from PIL import Image

from .denoise import normalize_percentile


def save_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    channel: int = 1,
    color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    """Save a quick RGB overlay for checking whether masks sit on cilia.

    The overlay is intentionally simple: the chosen grayscale channel becomes the
    background, and positive mask pixels are painted red. It is not used for
    training; it is a sanity-check artifact for humans.
    """
    bg = (normalize_percentile(image[channel]) * 255).astype("uint8")
    rgb = np.stack([bg, bg, bg], axis=-1)
    rgb[mask.astype(bool)] = color
    Image.fromarray(rgb).save(out_path)

