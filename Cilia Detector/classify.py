from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure, morphology

from .denoise import normalize_percentile


def classify_objects(
    stack: np.ndarray,
    mask: np.ndarray,
    min_area: int = 8,
    min_length_px: float = 5.0,
    min_ch2_mean: float = 0.10,
    min_ch1_support: float = 0.05,
) -> pd.DataFrame:
    """Measure and classify connected cilium candidates."""
    labeled = measure.label(mask.astype(bool))
    ch1 = normalize_percentile(stack[0])
    ch2 = normalize_percentile(stack[1])
    kif3b = normalize_percentile(stack[3]) if stack.shape[0] > 3 else None
    kif3a = normalize_percentile(stack[4]) if stack.shape[0] > 4 else None

    rows = []
    for region in measure.regionprops(labeled):
        if region.area < min_area:
            continue
        obj = labeled == region.label
        skel = morphology.skeletonize(obj)
        length_px = float(skel.sum())
        ch1_mean = float(ch1[obj].mean())
        ch2_mean = float(ch2[obj].mean())
        row = {
            "object_id": region.label,
            "area_px": int(region.area),
            "length_px": length_px,
            "centroid_y": float(region.centroid[0]),
            "centroid_x": float(region.centroid[1]),
            "bbox_min_y": int(region.bbox[0]),
            "bbox_min_x": int(region.bbox[1]),
            "bbox_max_y": int(region.bbox[2]),
            "bbox_max_x": int(region.bbox[3]),
            "ch1_axoneme_mean": ch1_mean,
            "ch2_arl13b_mean": ch2_mean,
            "kif3a_mean": float(kif3a[obj].mean()) if kif3a is not None else np.nan,
            "kif3b_mean": float(kif3b[obj].mean()) if kif3b is not None else np.nan,
        }
        row["is_likely_cilium"] = (
            row["area_px"] >= min_area
            and length_px >= min_length_px
            and ch2_mean >= min_ch2_mean
            and ch1_mean >= min_ch1_support
        )
        rows.append(row)
    return pd.DataFrame(rows)

