from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from skimage import measure, morphology


def _skeleton_path_metrics(skeleton: np.ndarray) -> tuple[float, float, float]:
    """Return skeleton length, end-to-end distance, and tortuosity for one object.

    Tortuosity is used here as a simple curvature proxy: a straight cilium is
    close to 1, while a more curved path is greater than 1.
    """
    coords = [tuple(point) for point in np.argwhere(skeleton)]
    if len(coords) < 2:
        return float(len(coords)), np.nan, np.nan

    points = set(coords)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in coords:
        neighbors[(y, x)] = [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy or dx) and (y + dy, x + dx) in points
        ]

    endpoints = [point for point, adjacent in neighbors.items() if len(adjacent) == 1]
    candidates = endpoints if len(endpoints) >= 2 else coords
    start, end = max(
        ((first, second) for index, first in enumerate(candidates) for second in candidates[index + 1 :]),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
    )

    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    while queue and end not in previous:
        point = queue.popleft()
        for adjacent in neighbors[point]:
            if adjacent not in previous:
                previous[adjacent] = point
                queue.append(adjacent)
    if end not in previous:
        return float(len(coords)), np.nan, np.nan

    path = [end]
    while path[-1] != start:
        parent = previous[path[-1]]
        if parent is None:
            break
        path.append(parent)
    path.reverse()
    path_length = sum(
        float(np.hypot(next_y - y, next_x - x))
        for (y, x), (next_y, next_x) in zip(path[:-1], path[1:])
    )
    direct_distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    tortuosity = path_length / direct_distance if direct_distance > 0 else np.nan
    return path_length, direct_distance, tortuosity


def measure_cilia_morphology(mask: np.ndarray, min_area: int = 8) -> pd.DataFrame:
    """Measure connected cilia objects from a binary mask only."""
    labels = measure.label(mask.astype(bool))
    rows: list[dict[str, float | int]] = []
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        object_mask = labels == region.label
        skeleton = morphology.skeletonize(object_mask)
        path_length, endpoint_distance, tortuosity = _skeleton_path_metrics(skeleton)
        rows.append(
            {
                "object_id": int(region.label),
                "area_px": int(region.area),
                "length_px": float(skeleton.sum()),
                "skeleton_path_length_px": path_length,
                "endpoint_distance_px": endpoint_distance,
                "curvature_tortuosity": tortuosity,
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
            }
        )
    return pd.DataFrame(rows)
