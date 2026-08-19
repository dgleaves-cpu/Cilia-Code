from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct
from typing import Iterable

import numpy as np
import tifffile as tiff
from skimage import draw, morphology


CHANNELS = {
    "axoneme": 0,
    "arl13b": 1,
    "dapi": 2,
    "kif3b": 3,
    "kif3a": 4,
}


ROI_TYPES = {
    0: "polygon",
    1: "rect",
    2: "oval",
    3: "line",
    4: "freeline",
    5: "polyline",
    7: "freehand",
    10: "point",
}


@dataclass
class ImageJRoi:
    roi_type: str
    top: int
    left: int
    bottom: int
    right: int
    channel: int | None = None
    name: str | None = None
    points: np.ndarray | None = None
    stroke_width: int = 1


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
    elif axes == "ZYX":
        pass
    elif stack.shape[-1] <= 8:
        stack = np.moveaxis(stack, -1, 0)
    else:
        raise ValueError(f"Unsupported TIFF axes {axes!r} with shape {stack.shape}")

    return stack.astype(np.uint16, copy=False), metadata


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    tiff.imwrite(path, mask.astype(np.uint8), imagej=True)


def write_stack(path: str | Path, stack: np.ndarray) -> None:
    tiff.imwrite(path, stack.astype(np.float32), imagej=True)


def extract_imagej_rois(metadata: dict) -> list[ImageJRoi]:
    """Extract common ImageJ overlay ROI shapes embedded in a TIFF.

    This minimal parser supports the ROI shapes produced by typical FIJI line,
    polygon, freehand, rectangle, and oval annotations. It avoids a hard runtime
    dependency on the external roifile package so the project is easy to run on
    Colab and clusters.
    """
    roi_blobs: Iterable[bytes] = metadata.get("Overlays", [])
    rois: list[ImageJRoi] = []
    for blob in roi_blobs:
        roi = _parse_roi(blob)
        if roi is not None:
            rois.append(roi)
    single_roi = metadata.get("ROI")
    if single_roi:
        roi = _parse_roi(single_roi)
        if roi is not None:
            rois.append(roi)
    return rois


def rois_to_mask(
    rois: Iterable[ImageJRoi],
    shape: tuple[int, int],
    channel: int | None = 2,
    line_width: int = 5,
) -> np.ndarray:
    """Rasterize ImageJ ROIs into a binary mask.

    ImageJ channel positions are 1-based. Pass `channel=None` to use all ROIs.
    """
    mask = np.zeros(shape, dtype=bool)
    for roi in rois:
        if channel is not None and roi.channel != channel:
            continue
        _draw_roi(mask, roi, line_width=line_width)
    return morphology.remove_small_objects(mask, min_size=8).astype(np.uint8)


def _parse_roi(blob: bytes) -> ImageJRoi | None:
    if len(blob) < 64 or blob[:4] != b"Iout":
        return None

    roi_type_code = blob[6]
    roi_type = ROI_TYPES.get(roi_type_code, f"type_{roi_type_code}")
    top, left, bottom, right, n = struct.unpack(">hhhhh", blob[8:18])
    stroke_width = max(1, _u16(blob, 34) if len(blob) >= 36 else 1)
    position = _u32(blob, 56) if len(blob) >= 60 else 0
    channel = position if position > 0 else None

    points = None
    if roi_type in {"polygon", "freehand", "polyline", "freeline", "point"} and n > 0:
        x0 = 64
        y0 = 64 + 2 * n
        if len(blob) >= y0 + 2 * n:
            xs = np.frombuffer(blob[x0 : x0 + 2 * n], dtype=">i2").astype(np.int32) + left
            ys = np.frombuffer(blob[y0 : y0 + 2 * n], dtype=">i2").astype(np.int32) + top
            points = np.column_stack([ys, xs])

    name = _parse_roi_name(blob)
    name_channel = _channel_from_name(name)
    if name_channel is not None:
        channel = name_channel

    return ImageJRoi(
        roi_type=roi_type,
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        channel=channel,
        name=name,
        points=points,
        stroke_width=stroke_width,
    )


def _draw_roi(mask: np.ndarray, roi: ImageJRoi, line_width: int) -> None:
    h, w = mask.shape
    if roi.roi_type == "rect":
        rr, cc = draw.rectangle(
            start=(max(0, roi.top), max(0, roi.left)),
            end=(min(h - 1, roi.bottom), min(w - 1, roi.right)),
            shape=mask.shape,
        )
        mask[rr, cc] = True
    elif roi.roi_type == "oval":
        cy = (roi.top + roi.bottom) / 2
        cx = (roi.left + roi.right) / 2
        ry = max(1, (roi.bottom - roi.top) / 2)
        rx = max(1, (roi.right - roi.left) / 2)
        rr, cc = draw.ellipse(cy, cx, ry, rx, shape=mask.shape)
        mask[rr, cc] = True
    elif roi.points is not None and len(roi.points) > 1:
        if roi.roi_type in {"polygon", "freehand"}:
            rr, cc = draw.polygon(roi.points[:, 0], roi.points[:, 1], shape=mask.shape)
            mask[rr, cc] = True
        else:
            roi_mask = np.zeros_like(mask)
            width = max(line_width, roi.stroke_width)
            for p0, p1 in zip(roi.points[:-1], roi.points[1:]):
                rr, cc = draw.line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
                rr = np.clip(rr, 0, h - 1)
                cc = np.clip(cc, 0, w - 1)
                roi_mask[rr, cc] = True
            mask |= morphology.binary_dilation(roi_mask, morphology.disk(width // 2))
    elif roi.roi_type == "line":
        roi_mask = np.zeros_like(mask)
        rr, cc = draw.line(roi.top, roi.left, roi.bottom, roi.right)
        rr = np.clip(rr, 0, h - 1)
        cc = np.clip(cc, 0, w - 1)
        roi_mask[rr, cc] = True
        mask |= morphology.binary_dilation(roi_mask, morphology.disk(max(1, line_width // 2)))


def _u16(blob: bytes, offset: int) -> int:
    return struct.unpack(">H", blob[offset : offset + 2])[0]


def _u32(blob: bytes, offset: int) -> int:
    return struct.unpack(">I", blob[offset : offset + 4])[0]


def _parse_roi_name(blob: bytes) -> str | None:
    if len(blob) < 64:
        return None
    header2 = _u32(blob, 60)
    if header2 <= 0 or len(blob) < header2 + 20:
        return None
    name_offset = _u32(blob, header2 + 16)
    name_length = _u32(blob, header2 + 20)
    if name_offset <= 0 or name_length <= 0:
        return None
    end = name_offset + 2 * name_length
    if end > len(blob):
        return None
    try:
        return blob[name_offset:end].decode("utf-16-be", errors="ignore")
    except UnicodeDecodeError:
        return None


def _channel_from_name(name: str | None) -> int | None:
    if not name:
        return None
    match = re.search(r"(?:ch|c|channel)[^\d]*(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None
