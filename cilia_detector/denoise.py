from __future__ import annotations

import numpy as np
from skimage import exposure, filters, morphology, restoration


def normalize_percentile(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.8,
    eps: float = 1e-6,
) -> np.ndarray:
    lo, hi = np.percentile(image, (lower, upper))
    image = (image.astype(np.float32) - lo) / max(float(hi - lo), eps)
    return np.clip(image, 0.0, 1.0)


def denoise_marker_channel(
    image: np.ndarray,
    background_radius: int = 32,
    nlm_patch_size: int = 5,
    nlm_patch_distance: int = 6,
    nlm_h: float = 0.08,
) -> np.ndarray:
    """Background-correct and denoise a fluorescence channel.

    The order is percentile normalization, rolling-ball-like background removal,
    non-local means denoising, and light contrast stretching. This is a strong
    classical baseline before training a learned denoiser.
    """
    x = normalize_percentile(image)
    background = filters.gaussian(x, sigma=max(1, background_radius / 6), preserve_range=True)
    x = np.clip(x - background, 0, None)
    if np.max(x) > 0:
        x = x / np.max(x)
    denoised = restoration.denoise_nl_means(
        x,
        h=nlm_h,
        patch_size=nlm_patch_size,
        patch_distance=nlm_patch_distance,
        fast_mode=True,
        preserve_range=True,
        channel_axis=None,
    )
    denoised = exposure.rescale_intensity(denoised, in_range=(0, np.percentile(denoised, 99.8)))
    return denoised.astype(np.float32)


def denoise_single_channel_input(stack: np.ndarray, channel_index: int = 1) -> np.ndarray:
    """Return one denoised model input channel with shape (1, H, W).

    The project convention is zero-based channel indexing, so channel_index=1
    means the second biological image channel.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected a channel-first stack with shape (C, H, W), got {stack.shape}")
    if channel_index < 0 or channel_index >= stack.shape[0]:
        raise ValueError(f"Requested channel index {channel_index}, but stack has {stack.shape[0]} channels")
    channel = denoise_marker_channel(stack[channel_index], background_radius=28, nlm_h=0.08)
    return channel[np.newaxis, ...].astype(np.float32)


def clean_binary_mask(mask: np.ndarray, min_size: int = 12) -> np.ndarray:
    mask = mask.astype(bool)
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.binary_closing(mask, morphology.disk(1))
    return mask.astype(np.uint8)
