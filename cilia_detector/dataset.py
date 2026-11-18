from __future__ import annotations

from pathlib import Path

import numpy as np

def save_patch_npz(out_path: str | Path, image: np.ndarray, mask: np.ndarray, source: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, image=image, mask=mask, source=source)
