from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile as tiff


def read_stack(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read an ImageJ/OME TIFF as a channel-first uint16 stack."""
    with tiff.TiffFile(path) as tif:
        stack = tif.asarray()
        metadata = tif.imagej_metadata or {}
        axes = tif.series[0].axes

    stack = np.asarray(stack)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    elif axes.endswith("CYX"):
        pass
    # Some acquisition software writes the channel/page dimension as ImageJ's
    # generic ``I`` axis; its array layout is still channel-first.
    elif axes == "IYX":
        pass
    elif axes == "ZYX":
        pass
    elif stack.shape[-1] <= 8:
        stack = np.moveaxis(stack, -1, 0)
    else:
        raise ValueError(f"Unsupported TIFF axes {axes!r} with shape {stack.shape}")

    return stack.astype(np.uint16, copy=False), metadata


def read_prediction_stacks(path: str | Path) -> list[np.ndarray]:
    """Read each channel-first image stored in a TIFF prediction upload.

    A regular CYX, IYX, or ZYX TIFF returns one stack. TIFF series with an
    explicit time or scene axis (for example TCYX) return one stack per image.
    The channel axis remains intact so callers can select channel index 1.
    """
    with tiff.TiffFile(path) as tif:
        stacks: list[np.ndarray] = []
        for series in tif.series:
            stacks.extend(_split_prediction_series(np.asarray(series.asarray()), series.axes))
    if not stacks:
        raise ValueError("The TIFF does not contain any image data.")
    return stacks


def _split_prediction_series(array: np.ndarray, axes: str) -> list[np.ndarray]:
    array = np.asarray(array)
    if array.ndim == 2:
        return [array[np.newaxis, ...].astype(np.uint16, copy=False)]
    if array.ndim != len(axes):
        raise ValueError(f"TIFF axes {axes!r} do not match array shape {array.shape}.")

    channel_candidates = [index for index, axis in enumerate(axes) if axis in {"C", "I", "Z"}]
    y_axis = axes.find("Y")
    x_axis = axes.find("X")
    if len(channel_candidates) != 1 or y_axis < 0 or x_axis < 0:
        raise ValueError(
            f"Unsupported TIFF axes {axes!r} with shape {array.shape}. "
            "Use a TIFF with one C, I, or Z channel axis and Y/X image axes."
        )

    channel_axis = channel_candidates[0]
    image_axes = [index for index in range(array.ndim) if index not in {channel_axis, y_axis, x_axis}]
    reordered = np.moveaxis(array, [*image_axes, channel_axis, y_axis, x_axis], range(array.ndim))
    image_count = int(np.prod(reordered.shape[: len(image_axes)])) if image_axes else 1
    stacks = reordered.reshape(image_count, *reordered.shape[len(image_axes) :])
    return [stack.astype(np.uint16, copy=False) for stack in stacks]


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    tiff.imwrite(path, mask.astype(np.uint8), imagej=True)


def write_stack(path: str | Path, stack: np.ndarray) -> None:
    tiff.imwrite(path, stack.astype(np.float32), imagej=True)
