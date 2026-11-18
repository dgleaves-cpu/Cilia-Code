#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import errno
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from skimage import measure, morphology
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cilia_detector.channel2_pipeline import (
    CHANNEL_INDEX,
    IMAGE_SIZE,
    MODEL_PATH,
    SUPPORTED_IMAGE_EXTENSIONS,
    find_image_mask_pairs,
    load_unet_model,
    normalize_selected_channel,
    png_bytes_from_mask,
    png_bytes_from_overlay,
    predict_preprocessed,
    preprocess_image_array,
    read_array,
    select_channel_preserve_dim,
)
from cilia_detector.dataset import save_patch_npz
from cilia_detector.denoise import denoise_single_channel_input
from cilia_detector.io import read_prediction_stacks, read_stack, write_mask
from cilia_detector.model import UNet, dice_loss
from cilia_detector.morphology import measure_cilia_morphology


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "app"
OPTIMIZATION_OUTPUT_ROOT = OUTPUT_ROOT / "optimization"
CHECKPOINT_ROOT = ROOT / "checkpoints"
SUPPORTED_ZIP_EXTENSIONS = {".zip"}
MASK_COLORMAP = ListedColormap(["#ffffff", "#d62728"])
FALSE_POSITIVE_COLOR = "#0072b2"
FALSE_NEGATIVE_COLOR = "#7b2cbf"
TRUE_POSITIVE_COLOR = "#2e7d32"


def choose_default_model_path() -> Path:
    preferred = [
        ROOT / MODEL_PATH,
        CHECKPOINT_ROOT / "channel2_uploaded_cilia_unet.pt",
        CHECKPOINT_ROOT / "cilia_unet.pt",
    ]
    checkpoints = preferred + (sorted(CHECKPOINT_ROOT.glob("*.pt")) if CHECKPOINT_ROOT.exists() else [])
    seen: set[Path] = set()
    for path in checkpoints:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if checkpoint.get("in_channels") == 1 and checkpoint.get("channel_index") == CHANNEL_INDEX:
            return path
    return ROOT / MODEL_PATH


DEFAULT_MODEL_PATH = choose_default_model_path()

app = FastAPI(
    title="Cilia Detector App",
    description="Browser app for channel-2 cilia U-Net training, testing, prediction, and mask analysis.",
    version="0.3.0",
)

# Jobs stay in process because this is a single-user local desktop application.
# A lock prevents two simultaneous U-Net runs from competing for laptop memory.
TRAINING_JOBS: dict[str, dict[str, object]] = {}
TRAINING_JOBS_LOCK = threading.Lock()
TRAINING_RUN_LOCK = threading.Lock()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_output_dir(prefix: str) -> Path:
    out_dir = OUTPUT_ROOT / f"{prefix}_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def create_training_job(kind: str, epochs: int) -> str:
    job_id = uuid.uuid4().hex
    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "phase": "Waiting to start",
            "message": "Training request received.",
            "progress_percent": 0,
            "completed_epochs": 0,
            "total_epochs": epochs,
            "last_metrics": None,
            "started_at": time.time(),
            "finished_at": None,
            "result": None,
            "error": None,
        }
    return job_id


def update_training_job(job_id: str, **changes: object) -> None:
    with TRAINING_JOBS_LOCK:
        if job_id in TRAINING_JOBS:
            TRAINING_JOBS[job_id].update(changes)


def training_job_snapshot(job_id: str) -> dict[str, object]:
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found. Start training again from the app.")
        snapshot = dict(job)
    elapsed = (snapshot.get("finished_at") or time.time()) - float(snapshot["started_at"])
    snapshot["elapsed_seconds"] = round(elapsed, 1)
    return snapshot


def resolve_project_path(value: str | None, default: Path | None = None) -> Path:
    if not value:
        if default is None:
            raise ValueError("A path is required.")
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def relative_url(path: Path) -> str:
    path = path.resolve()
    try:
        rel = path.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Output path is outside {OUTPUT_ROOT}: {path}") from exc
    return f"/outputs/{rel.as_posix()}"


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise HTTPException(status_code=400, detail="The ZIP contains an unsafe path. Remove absolute paths or ../ entries.")
        zf.extractall(destination)


def find_training_dirs(extract_dir: Path) -> tuple[Path, Path]:
    """Locate a paired dataset using the app's common training/test folder names."""
    candidates = [extract_dir]
    candidates.extend([p for p in extract_dir.iterdir() if p.is_dir()])

    def child_dir_case_insensitive(base: Path, name: str) -> Path | None:
        exact = base / name
        if exact.is_dir():
            return exact
        for child in base.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
        return None

    folder_pairs = [
        ("images", "masks"),
        ("training_images", "training_masks"),
        ("test_images", "test_masks"),
    ]
    for base in candidates:
        for image_name, mask_name in folder_pairs:
            image_dir = child_dir_case_insensitive(base, image_name)
            mask_dir = child_dir_case_insensitive(base, mask_name)
            if image_dir is not None and mask_dir is not None:
                return image_dir, mask_dir
    raise HTTPException(
        status_code=400,
        detail=(
            "Could not find paired image and mask folders. Use images/ and masks/, "
            "training_images/ and training_masks/, or test_images/ and test_masks/. "
            "Each pair must have matching filename stems, for example sample_001.tif in both folders."
        ),
    )


def find_mask_files(extract_dir: Path) -> list[Path]:
    """Find standalone cilia masks in the common ZIP layouts used by the app."""
    mask_folder_names = {"masks", "training_masks", "test_masks"}

    def is_archive_metadata(path: Path) -> bool:
        relative_parts = path.relative_to(extract_dir).parts
        return any(part == "__MACOSX" or part.startswith("._") or part == ".DS_Store" for part in relative_parts)

    def is_supported_mask_file(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not is_archive_metadata(path)

    mask_dirs = sorted(
        path
        for path in extract_dir.rglob("*")
        if path.is_dir() and path.name.lower() in mask_folder_names and not is_archive_metadata(path)
    )
    if mask_dirs:
        files = sorted(
            path
            for mask_dir in mask_dirs
            for path in mask_dir.rglob("*")
            if is_supported_mask_file(path)
        )
    else:
        # Also allow a ZIP that contains only mask files, optionally inside one wrapper folder.
        files = sorted(
            path
            for path in extract_dir.rglob("*")
            if is_supported_mask_file(path)
        )
    if not files:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=(
                "No mask image files were found. Upload a ZIP containing masks/ (or training_masks/ or "
                f"test_masks/) with binary mask files. Supported formats: {supported}."
            ),
        )
    return files


def save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)


def validate_zip_upload(upload: UploadFile) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_ZIP_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")


def validate_image_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format {suffix or '(none)'}. Supported formats: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}.",
        )
    return suffix


def overlay_file_from_channel(channel_hwc: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    out_path.write_bytes(png_bytes_from_overlay(channel_hwc, mask))


def mask_png_file(mask: np.ndarray, out_path: Path) -> None:
    out_path.write_bytes(png_bytes_from_mask(mask.astype(np.uint8)))


def full_resolution_prediction(
    model: UNet,
    image_path: Path,
    threshold: float,
    tile_size: int,
    overlap: int,
    min_object_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    stack, _ = read_stack(image_path)
    return full_resolution_prediction_from_stack(
        model,
        stack,
        threshold=threshold,
        tile_size=tile_size,
        overlap=overlap,
        min_object_size=min_object_size,
        source_name=image_path.name,
    )


def full_resolution_prediction_from_stack(
    model: UNet,
    stack: np.ndarray,
    threshold: float,
    tile_size: int,
    overlap: int,
    min_object_size: int,
    source_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    selected = denoise_single_channel_input(stack, channel_index=CHANNEL_INDEX)
    require_channel2_nchw(selected, context=f"{source_name} prediction")
    probability = predict_tiled(model, selected, get_device(), tile_size=tile_size, overlap=overlap)
    mask = mask_from_probability(probability, threshold, min_object_size)
    channel2_hwc = np.moveaxis(selected, 0, -1)
    objects = measure_cilia_morphology(mask, min_area=min_object_size)
    return probability, mask, channel2_hwc, objects


def prediction_stacks_from_file(image_path: Path) -> list[np.ndarray]:
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        return read_prediction_stacks(image_path)
    return [read_array(image_path)]


def prediction_output_stem(original_name: str, image_index: int, image_count: int, used_stems: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem).strip("._-") or "image"
    if image_count > 1:
        base = f"{base}_image_{image_index:03d}"
    candidate = base
    suffix = 2
    while candidate in used_stems:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_stems.add(candidate)
    return candidate


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def predict_tiled(model: UNet, inputs: np.ndarray, device: torch.device, tile_size: int = 512, overlap: int = 64) -> np.ndarray:
    require_channel2_nchw(inputs, context="tiled prediction")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be at least 0 and smaller than tile_size.")
    _, h, w = inputs.shape
    if h < 32 or w < 32:
        raise ValueError(f"Image is too small for U-Net prediction: {h}x{w}.")

    tile_h = min(tile_size, h)
    tile_w = min(tile_size, w)
    stride_y = max(1, tile_h - min(overlap, tile_h - 1))
    stride_x = max(1, tile_w - min(overlap, tile_w - 1))
    prob = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for y in range(0, h, stride_y):
            for x in range(0, w, stride_x):
                y0 = min(y, max(0, h - tile_h))
                x0 = min(x, max(0, w - tile_w))
                tile = inputs[:, y0 : y0 + tile_h, x0 : x0 + tile_w]
                tensor = torch.from_numpy(tile[None, ...]).float().to(device)
                pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                prob[y0 : y0 + tile_h, x0 : x0 + tile_w] += pred
                weight[y0 : y0 + tile_h, x0 : x0 + tile_w] += 1
    return prob / np.maximum(weight, 1e-6)


def binary_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = float(np.logical_and(pred, true).sum())
    pred_sum = float(pred.sum())
    true_sum = float(true.sum())
    union = float(np.logical_or(pred, true).sum())
    eps = 1e-6
    return {
        "dice": (2 * intersection + eps) / (pred_sum + true_sum + eps),
        "iou": (intersection + eps) / (union + eps),
        "predicted_pixels": pred_sum,
        "true_pixels": true_sum,
    }


def mask_from_probability(probability: np.ndarray, threshold: float, min_object_size: int) -> np.ndarray:
    mask = probability >= threshold
    return morphology.remove_small_objects(mask, min_size=min_object_size).astype(np.uint8)


def object_detection_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    iou_threshold: float,
    min_object_size: int,
) -> dict[str, float]:
    """Greedily match connected predicted and manual cilia by object IoU."""
    pred_labels = measure.label(pred_mask.astype(bool))
    true_clean = morphology.remove_small_objects(true_mask.astype(bool), min_size=min_object_size)
    true_labels = measure.label(true_clean)
    pred_count = int(pred_labels.max())
    true_count = int(true_labels.max())
    if not pred_count or not true_count:
        tp = 0
    else:
        overlap = (true_labels > 0) & (pred_labels > 0)
        true_area = np.bincount(true_labels.ravel(), minlength=true_count + 1)
        pred_area = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)
        codes = true_labels[overlap] * (pred_count + 1) + pred_labels[overlap]
        codes, intersections = np.unique(codes, return_counts=True)
        candidates = []
        for code, intersection in zip(codes, intersections):
            true_id = int(code // (pred_count + 1))
            pred_id = int(code % (pred_count + 1))
            union = true_area[true_id] + pred_area[pred_id] - intersection
            iou_value = float(intersection / union) if union else 0.0
            if iou_value >= iou_threshold:
                candidates.append((iou_value, true_id, pred_id))
        matched_true: set[int] = set()
        matched_pred: set[int] = set()
        tp = 0
        for _, true_id, pred_id in sorted(candidates, reverse=True):
            if true_id not in matched_true and pred_id not in matched_pred:
                matched_true.add(true_id)
                matched_pred.add(pred_id)
                tp += 1
    fp = pred_count - tp
    fn = true_count - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "object_true_count": true_count,
        "object_predicted_count": pred_count,
        "object_true_positive": tp,
        "object_false_positive": fp,
        "object_false_negative": fn,
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1,
    }


def sample_pixels_for_precision_recall(
    probability: np.ndarray,
    true_mask: np.ndarray,
    rng: np.random.Generator,
    max_positive: int = 50_000,
    max_negative: int = 500_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted pixel samples that retain the original class prevalence."""
    scores = probability.reshape(-1)
    truth = true_mask.reshape(-1).astype(bool)
    positive = np.flatnonzero(truth)
    negative = np.flatnonzero(~truth)
    if len(positive) == 0 or len(negative) == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.float32), np.array([], dtype=np.float64)
    sampled_positive = rng.choice(positive, size=min(len(positive), max_positive), replace=False)
    sampled_negative = rng.choice(negative, size=min(len(negative), max_negative), replace=False)
    indices = np.concatenate([sampled_positive, sampled_negative])
    labels = np.concatenate([np.ones(len(sampled_positive), dtype=np.uint8), np.zeros(len(sampled_negative), dtype=np.uint8)])
    weights = np.concatenate([
        np.full(len(sampled_positive), len(positive) / len(sampled_positive), dtype=np.float64),
        np.full(len(sampled_negative), len(negative) / len(sampled_negative), dtype=np.float64),
    ])
    order = rng.permutation(len(indices))
    return labels[order], scores[indices][order].astype(np.float32), weights[order]


def save_precision_recall_plot(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, out_dir: Path) -> dict[str, object]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"pr_note": "Precision-recall/AP could not be calculated because the test masks did not contain both cilia and background pixels."}
    precision, recall, _ = precision_recall_curve(labels, scores, sample_weight=weights)
    average_precision = float(average_precision_score(labels, scores, sample_weight=weights))
    prevalence = float(np.average(labels, weights=weights))
    plot_path = out_dir / "pixel_precision_recall.png"
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(recall, precision, color="#376c3c", linewidth=2, label=f"Pixel PR (AP = {average_precision:.3f})")
    ax.axhline(prevalence, linestyle="--", color="#707070", label=f"Cilia prevalence ({prevalence:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Pixel Precision-Recall on Test Masks", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return {
        "pixel_average_precision": average_precision,
        "pr_sample_pixels": int(len(labels)),
        "pr_plot_url": relative_url(plot_path),
        "pr_note": "Precision-recall/AP uses weighted pixel sampling to preserve the test set's cilia prevalence.",
    }


def save_threshold_sweep_plot(rows: list[dict[str, float]], selected_threshold: float, out_dir: Path) -> dict[str, object]:
    table = pd.DataFrame(rows)
    csv_path = out_dir / "threshold_sweep.csv"
    table.to_csv(csv_path, index=False)
    if table.empty:
        return {"threshold_sweep_csv_url": relative_url(csv_path)}
    best = table.loc[table["dice"].idxmax()]
    plot_path = out_dir / "threshold_sweep.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["threshold"], table["dice"], marker="o", color="#166b8f", label="Dice")
    ax.plot(table["threshold"], table["iou"], marker="o", color="#b35a2a", label="IoU")
    ax.axvline(selected_threshold, linestyle="--", color="#707070", label=f"Selected ({selected_threshold:.2f})")
    ax.axvline(float(best["threshold"]), linestyle=":", color="#376c3c", label=f"Best Dice ({best['threshold']:.2f})")
    ax.set(xlabel="Prediction threshold", ylabel="Score", title="Threshold Sweep on Test Masks", xlim=(0.05, 0.95), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return {
        "threshold_sweep_csv_url": relative_url(csv_path),
        "threshold_sweep_plot_url": relative_url(plot_path),
        "best_dice_threshold": float(best["threshold"]),
        "best_dice_from_sweep": float(best["dice"]),
    }


def save_failure_review_gallery(records: list[dict[str, object]], out_dir: Path, limit: int = 10) -> dict[str, object]:
    """Save low-Dice cases with Channel 2 and red-on-white mask error callouts."""
    gallery_dir = out_dir / "failure_review"
    gallery_dir.mkdir(exist_ok=True)
    review = []
    for record in sorted(records, key=lambda item: float(item["dice"]))[:limit]:
        stem = str(record["stem"])
        path = gallery_dir / f"{stem}_failure_review.png"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        channel = np.asarray(record["channel2"])[..., 0]
        true_mask = np.asarray(record["true_mask"])
        pred_mask = np.asarray(record["pred_mask"])
        circles = object_review_circles(pred_mask, true_mask)

        axes[0].imshow(channel, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Input")
        axes[1].imshow(true_mask, cmap=MASK_COLORMAP, vmin=0, vmax=1)
        axes[1].set_title("True Cilia Mask")
        axes[2].imshow(pred_mask, cmap=MASK_COLORMAP, vmin=0, vmax=1)
        axes[2].set_title("Predicted Cilia Mask")
        draw_object_circles(axes[2], circles, kind="true_positive")
        draw_object_circles(axes[2], circles, kind="false_positive")
        draw_object_circles(axes[2], circles, kind="false_negative")
        axes[0].axis("off")
        for axis in axes[1:]:
            style_mask_axis(axis)
        add_object_legend(fig, circles)
        fig.suptitle(f"{stem}: Dice {float(record['dice']):.3f}")
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        review.append({"stem": stem, "dice": float(record["dice"]), "url": relative_url(path)})
    return {
        "failure_review": review,
        "failure_review_note": "Failure review is ordered from lowest to highest Dice; inspect these cases first when improving labels or training data.",
    }


def object_review_circles(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    iou_threshold: float = 0.10,
) -> list[dict[str, float | str]]:
    """Return callout circles for matched and unmatched cilia objects."""
    pred_labels = measure.label(pred_mask.astype(bool))
    true_labels = measure.label(true_mask.astype(bool))
    pred_count = int(pred_labels.max())
    true_count = int(true_labels.max())
    pred_area = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)
    true_area = np.bincount(true_labels.ravel(), minlength=true_count + 1)
    overlap = (pred_labels > 0) & (true_labels > 0)
    matched_pred: set[int] = set()
    matched_true: set[int] = set()
    if overlap.any():
        codes, intersections = np.unique(
            true_labels[overlap] * (pred_count + 1) + pred_labels[overlap],
            return_counts=True,
        )
        candidates = []
        for code, intersection in zip(codes, intersections):
            true_id = int(code // (pred_count + 1))
            pred_id = int(code % (pred_count + 1))
            union = true_area[true_id] + pred_area[pred_id] - intersection
            iou = float(intersection / union) if union else 0.0
            if iou >= iou_threshold:
                candidates.append((iou, true_id, pred_id))
        for _, true_id, pred_id in sorted(candidates, reverse=True):
            if true_id not in matched_true and pred_id not in matched_pred:
                matched_true.add(true_id)
                matched_pred.add(pred_id)

    circles: list[dict[str, float | str]] = []
    for region in measure.regionprops(pred_labels):
        kind = "true_positive" if region.label in matched_pred else "false_positive"
        min_row, min_col, max_row, max_col = region.bbox
        padding = 14 if kind == "true_positive" else 9
        minimum_radius = 18.0 if kind == "true_positive" else 14.0
        circles.append({
            "kind": kind,
            "row": float(region.centroid[0]),
            "col": float(region.centroid[1]),
            "radius": max(minimum_radius, float(np.hypot(max_row - min_row, max_col - min_col) / 2 + padding)),
        })
    for region in measure.regionprops(true_labels):
        if region.label in matched_true:
            continue
        kind = "false_negative"
        min_row, min_col, max_row, max_col = region.bbox
        circles.append({
            "kind": kind,
            "row": float(region.centroid[0]),
            "col": float(region.centroid[1]),
            "radius": max(14.0, float(np.hypot(max_row - min_row, max_col - min_col) / 2 + 9)),
        })
    return circles


def draw_object_circles(axis: plt.Axes, circles: list[dict[str, float | str]], kind: str) -> None:
    color = {
        "true_positive": TRUE_POSITIVE_COLOR,
        "false_positive": FALSE_POSITIVE_COLOR,
        "false_negative": FALSE_NEGATIVE_COLOR,
    }[kind]
    for item in circles:
        if item["kind"] != kind:
            continue
        axis.add_patch(
            Circle(
                (float(item["col"]), float(item["row"])),
                radius=float(item["radius"]),
                fill=False,
                edgecolor=color,
                linewidth=1.2,
                linestyle="-",
            )
        )


def style_mask_axis(axis: plt.Axes) -> None:
    """Keep a light frame around white mask panels while hiding plot controls."""
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(0.7)


def add_object_legend(fig: plt.Figure, circles: list[dict[str, float | str]]) -> None:
    present = {str(item["kind"]) for item in circles}
    handles = []
    if "true_positive" in present:
        handles.append(Line2D([0], [0], color=TRUE_POSITIVE_COLOR, linewidth=1.2, label="True-positive object"))
    if "false_positive" in present:
        handles.append(Line2D([0], [0], color=FALSE_POSITIVE_COLOR, linewidth=1.2, label="False-positive object"))
    if "false_negative" in present:
        handles.append(Line2D([0], [0], color=FALSE_NEGATIVE_COLOR, linewidth=1.2, label="False-negative object"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)


def save_training_plots(history: list[dict[str, float]], out_dir: Path) -> dict[str, str]:
    """Save loss and validation-metric plots for one training run."""
    table = pd.DataFrame(history)
    history_path = out_dir / "training_history.csv"
    table.to_csv(history_path, index=False)
    if table.empty:
        return {"history_csv_url": relative_url(history_path)}

    loss_path = out_dir / "training_loss.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["epoch"], table["train_loss"], marker="o", color="#166b8f", label="Training loss")
    ax.plot(table["epoch"], table["val_loss"], marker="o", color="#b35a2a", label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="BCE + Dice loss", title="Training and Validation Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    metric_path = out_dir / "validation_metrics.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["epoch"], table["val_dice"], marker="o", color="#376c3c", label="Validation Dice")
    ax.plot(table["epoch"], table["val_iou"], marker="o", color="#704c9b", label="Validation IoU")
    ax.set(xlabel="Epoch", ylabel="Score", title="Validation Segmentation Metrics", ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(metric_path, dpi=160)
    plt.close(fig)
    return {
        "history_csv_url": relative_url(history_path),
        "loss_plot_url": relative_url(loss_path),
        "validation_metrics_plot_url": relative_url(metric_path),
    }


def sample_pixels_for_roc(
    probability: np.ndarray,
    true_mask: np.ndarray,
    rng: np.random.Generator,
    max_per_class: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a balanced pixel sample so large TIFFs remain inexpensive to plot."""
    scores = probability.reshape(-1)
    truth = true_mask.reshape(-1).astype(bool)
    positive = np.flatnonzero(truth)
    negative = np.flatnonzero(~truth)
    if len(positive) == 0 or len(negative) == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.float32)
    count = min(len(positive), len(negative), max_per_class)
    positive = rng.choice(positive, size=count, replace=False)
    negative = rng.choice(negative, size=count, replace=False)
    indices = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(count, dtype=np.uint8), np.zeros(count, dtype=np.uint8)])
    order = rng.permutation(len(indices))
    return labels[order], scores[indices][order].astype(np.float32)


def save_roc_auc_plot(labels: np.ndarray, scores: np.ndarray, out_dir: Path) -> dict[str, object]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"roc_note": "ROC/AUC could not be calculated because the test masks did not contain both cilia and background pixels."}
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    plot_path = out_dir / "pixel_roc_auc.png"
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color="#166b8f", linewidth=2, label=f"Pixel ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#707070", label="Chance")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Pixel ROC/AUC on Test Masks", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return {
        "pixel_roc_auc": roc_auc,
        "roc_sample_pixels": int(len(labels)),
        "roc_plot_url": relative_url(plot_path),
        "roc_note": "ROC/AUC uses a balanced sample of labeled cilia and background pixels from the test set.",
    }


def save_morphology_tsne(morphology_table: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Save mask-only cilium measurements and two exploratory t-SNE views."""
    csv_path = out_dir / "cilia_morphology.csv"
    morphology_table.to_csv(csv_path, index=False)
    result: dict[str, object] = {"morphology_csv_url": relative_url(csv_path)}
    if morphology_table.empty:
        result["tsne_note"] = "No predicted cilia objects met the minimum object-size setting."
        return result
    feature_columns = ["length_px", "curvature_tortuosity", "area_px", "skeleton_path_length_px"]
    usable = morphology_table.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns).copy()
    if len(usable) < 4:
        result["tsne_note"] = "t-SNE needs at least 4 measured cilia with valid length and curvature values."
        return result

    features = usable[feature_columns].to_numpy(dtype=np.float32)
    non_constant = np.nanstd(features, axis=0) > 1e-8
    if int(non_constant.sum()) < 2:
        result["tsne_note"] = "t-SNE needs variation in at least two morphology measurements."
        return result
    features = StandardScaler().fit_transform(features[:, non_constant])
    perplexity = min(30, max(2, (len(usable) - 1) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(features)
    usable["tsne_1"] = embedding[:, 0]
    usable["tsne_2"] = embedding[:, 1]
    coordinates_path = out_dir / "cilia_morphology_tsne.csv"
    usable.to_csv(coordinates_path, index=False)
    result["tsne_coordinates_csv_url"] = relative_url(coordinates_path)

    for column, title, filename, cmap in [
        ("length_px", "t-SNE of Cilia Morphology: Length", "tsne_cilia_length.png", "viridis"),
        ("curvature_tortuosity", "t-SNE of Cilia Morphology: Curvature", "tsne_cilia_curvature.png", "plasma"),
    ]:
        plot_path = out_dir / filename
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        points = ax.scatter(usable["tsne_1"], usable["tsne_2"], c=usable[column], cmap=cmap, s=42, alpha=0.9)
        colorbar = fig.colorbar(points, ax=ax)
        colorbar.set_label("Length (pixels)" if column == "length_px" else "Curvature proxy (tortuosity)")
        ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=title)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        result["tsne_length_plot_url" if column == "length_px" else "tsne_curvature_plot_url"] = relative_url(plot_path)
    result["tsne_note"] = "t-SNE is exploratory: nearby points have similar length, curvature proxy, area, and skeleton path length."
    return result


def binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    targets = targets.bool()
    intersection = torch.logical_and(preds, targets).sum().float()
    pred_sum = preds.sum().float()
    target_sum = targets.sum().float()
    union = torch.logical_or(preds, targets).sum().float()
    eps = torch.tensor(1e-6, device=logits.device)
    return {
        "dice": float(((2 * intersection + eps) / (pred_sum + target_sum + eps)).item()),
        "iou": float(((intersection + eps) / (union + eps)).item()),
    }


def read_binary_mask_file(mask_path: Path) -> np.ndarray:
    arr = np.asarray(read_array(mask_path))
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[-1] <= 4:
            arr = arr[..., 0]
        elif arr.shape[0] <= 4:
            arr = arr[0]
        else:
            raise ValueError(f"Mask {mask_path.name} has ambiguous shape {arr.shape}. Use a 2D binary mask.")
    if arr.ndim != 2:
        raise ValueError(f"Mask {mask_path.name} should be 2D, got shape {arr.shape}.")
    return (arr > 0).astype(np.uint8)


def load_full_resolution_channel2_input(image_path: Path) -> np.ndarray:
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        stack, _ = read_stack(image_path)
        inputs = denoise_single_channel_input(stack, channel_index=CHANNEL_INDEX)
        require_channel2_nchw(inputs, context=f"{image_path.name} training input")
        return inputs
    image = read_array(image_path)
    channel_hwc = select_channel_preserve_dim(image, channel_index=CHANNEL_INDEX)
    channel_hwc = normalize_selected_channel(channel_hwc)
    inputs = np.moveaxis(channel_hwc, -1, 0).astype(np.float32)
    require_channel2_nchw(inputs, context=f"{image_path.name} training input")
    return inputs


def require_channel2_nchw(inputs: np.ndarray, context: str) -> None:
    if inputs.ndim != 3 or inputs.shape[0] != 1:
        raise ValueError(
            f"{context} must contain exactly one selected channel with shape (1, height, width), "
            f"but received {inputs.shape}. Delete old multi-channel patch files and recreate the dataset."
        )


def iter_training_patches_balanced(
    inputs: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    stride: int,
    min_positive_pixels: int,
    background_ratio: float,
    seed: int = 42,
):
    require_channel2_nchw(inputs, context="patch creation")
    _, height, width = inputs.shape
    positive: list[tuple[np.ndarray, np.ndarray, int]] = []
    background: list[tuple[np.ndarray, np.ndarray, int]] = []
    for y in range(0, max(1, height - patch_size + 1), stride):
        for x in range(0, max(1, width - patch_size + 1), stride):
            image_patch = inputs[:, y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]
            if image_patch.shape[-2:] != (patch_size, patch_size):
                continue
            positive_pixels = int(mask_patch.sum())
            item = (image_patch.astype(np.float32), mask_patch.astype(np.float32), positive_pixels)
            if positive_pixels >= min_positive_pixels:
                positive.append(item)
            else:
                background.append(item)

    rng = random.Random(seed)
    rng.shuffle(background)
    if positive:
        max_background = int(round(len(positive) * max(0.0, background_ratio)))
    else:
        max_background = min(len(background), 8)

    selected = positive + background[:max_background]
    rng.shuffle(selected)
    yield from selected


def create_paired_patch_dataset(
    image_dir: Path,
    mask_dir: Path,
    out_dir: Path,
    patch_size: int,
    stride: int,
    min_positive_pixels: int,
    background_ratio: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    pairs = find_image_mask_pairs(image_dir, mask_dir)
    if not pairs:
        raise HTTPException(status_code=400, detail="No paired image/mask files were found.")

    patch_dir = out_dir / "patches"
    overlay_dir = out_dir / "overlays"
    patch_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_patches = 0
    total_positive_patches = 0
    total_background_patches = 0
    for pair_index, pair in enumerate(pairs, start=1):
        inputs = load_full_resolution_channel2_input(pair.image_path)
        mask = read_binary_mask_file(pair.mask_path)
        if inputs.shape[-2:] != mask.shape:
            raise ValueError(
                f"{pair.stem}: image channel shape {inputs.shape[-2:]} does not match mask shape {mask.shape}. "
                "The mask must have the same height and width as the source image."
            )

        overlay_path = overlay_dir / f"{pair.stem}_channel2_mask_overlay.png"
        overlay_file_from_channel(np.moveaxis(inputs, 0, -1), mask, overlay_path)

        patch_count = 0
        positive_patch_count = 0
        background_patch_count = 0
        for image_patch, mask_patch, positive_pixels in iter_training_patches_balanced(
            inputs,
            mask,
            patch_size=patch_size,
            stride=stride,
            min_positive_pixels=min_positive_pixels,
            background_ratio=background_ratio,
        ):
            save_patch_npz(patch_dir / f"{pair.stem}_patch{patch_count:04d}.npz", image_patch, mask_patch, str(pair.image_path))
            patch_count += 1
            total_patches += 1
            if positive_pixels >= min_positive_pixels:
                positive_patch_count += 1
                total_positive_patches += 1
            else:
                background_patch_count += 1
                total_background_patches += 1

        rows.append(
            {
                "stem": pair.stem,
                "image": pair.image_path.name,
                "mask": pair.mask_path.name,
                "image_shape": str(tuple(inputs.shape)),
                "mask_shape": str(tuple(mask.shape)),
                "mask_pixels": int(mask.sum()),
                "patches": patch_count,
                "positive_patches": positive_patch_count,
                "background_patches": background_patch_count,
                "overlay_url": relative_url(overlay_path),
            }
        )
        if progress_callback is not None:
            progress_callback(pair_index, len(pairs))

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "paired_patch_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {
        "pair_count": len(pairs),
        "patch_count": total_patches,
        "positive_patch_count": total_positive_patches,
        "background_patch_count": total_background_patches,
        "patch_size": patch_size,
        "stride": stride,
        "min_positive_pixels": min_positive_pixels,
        "background_ratio": background_ratio,
        "patch_dir": str(patch_dir),
        "summary_path": str(summary_path),
        "summary_url": relative_url(summary_path),
        "pairs": rows,
    }


class PatchDataset(Dataset):
    def __init__(self, patch_dir: Path, preload: bool = False, augment: bool = True):
        self.paths = sorted(patch_dir.glob("*.npz"))
        if not self.paths:
            raise ValueError(f"No .npz patch files found in {patch_dir}.")
        self.augment = augment
        self.preloaded: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if preload:
            self.preloaded = []
            for path in self.paths:
                data = np.load(path)
                image = torch.from_numpy(data["image"]).float()
                if image.ndim != 3 or image.shape[0] != 1:
                    raise ValueError(
                        f"{path.name} contains {tuple(image.shape)} input data. This app only trains on "
                        "the second image channel and requires cached patches shaped (1, height, width). "
                        "Create a new patch dataset from the original multichannel images."
                    )
                mask = torch.from_numpy(data["mask"][None, ...]).float()
                self.preloaded.append((image, mask))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        if self.preloaded is None:
            data = np.load(self.paths[idx])
            image = torch.from_numpy(data["image"]).float()
            if image.ndim != 3 or image.shape[0] != 1:
                raise ValueError(
                    f"{self.paths[idx].name} is not a second-channel-only patch. Recreate the patch dataset."
                )
            mask = torch.from_numpy(data["mask"][None, ...]).float()
        else:
            image, mask = self.preloaded[idx]
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        return image, mask


def train_from_patch_dir(
    patch_dir: Path,
    model_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    overwrite_model: bool,
    epoch_callback: Callable[[int, int, dict[str, float] | None], None] | None = None,
    device: torch.device | None = None,
) -> list[dict[str, float]]:
    if model_path.exists() and not overwrite_model:
        raise HTTPException(status_code=400, detail=f"Refusing to overwrite {model_path}. Enable overwrite or choose a new path.")

    device = device or get_device()
    load_start = time.perf_counter()
    # Reading patches only when needed keeps large 512 px datasets from competing
    # with U-Net activations for RAM or unified Mac GPU memory.
    dataset = PatchDataset(patch_dir, preload=False, augment=True)
    print(f"Prepared {len(dataset)} patches for streaming in {time.perf_counter() - load_start:.1f}s on {device.type}", flush=True)
    if epoch_callback is not None:
        epoch_callback(0, epochs, None)
    if len(dataset) > 1 and val_fraction > 0:
        val_size = max(1, int(len(dataset) * val_fraction))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    else:
        train_ds = dataset
        val_ds = dataset

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = UNet(in_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0], device=device))

    best_val = float("inf")
    history: list[dict[str, float]] = []
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        dice_total = 0.0
        iou_total = 0.0
        seen = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                batch_n = images.size(0)
                val_loss += float(loss.item()) * batch_n
                metrics = binary_metrics_from_logits(logits, masks)
                dice_total += metrics["dice"] * batch_n
                iou_total += metrics["iou"] * batch_n
                seen += batch_n
        val_loss /= len(val_loader.dataset)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": dice_total / max(seen, 1),
            "val_iou": iou_total / max(seen, 1),
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(epoch, epochs, row)
        print(
            f"epoch={epoch:03d}/{epochs:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f} "
            f"val_iou={row['val_iou']:.4f} seconds={time.perf_counter() - epoch_start:.1f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_channels": 1,
                    "channel_index": CHANNEL_INDEX,
                    "threshold": 0.5,
                    "best_val_loss": best_val,
                },
                model_path,
            )
    return history


def is_broken_pipe_error(exc: Exception) -> bool:
    """Recognize the OS error reported when an MPS worker exits unexpectedly."""
    if isinstance(exc, BrokenPipeError):
        return True
    if isinstance(exc, OSError) and exc.errno == errno.EPIPE:
        return True
    return "broken pipe" in str(exc).lower()


def train_with_device_recovery(
    patch_dir: Path,
    model_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    overwrite_model: bool,
    epoch_callback: Callable[[int, int, dict[str, float] | None], None] | None,
    recovery_callback: Callable[[], None] | None = None,
) -> list[dict[str, float]]:
    """Train on the preferred device and safely retry on CPU after early MPS EPIPE."""
    primary_device = get_device()
    try:
        return train_from_patch_dir(
            patch_dir=patch_dir,
            model_path=model_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_fraction=val_fraction,
            overwrite_model=overwrite_model,
            epoch_callback=epoch_callback,
            device=primary_device,
        )
    except Exception as exc:
        no_checkpoint_saved = not model_path.exists()
        if primary_device.type != "mps" or not no_checkpoint_saved or not is_broken_pipe_error(exc):
            raise

        if recovery_callback is not None:
            recovery_callback()
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return train_from_patch_dir(
            patch_dir=patch_dir,
            model_path=model_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_fraction=val_fraction,
            overwrite_model=overwrite_model,
            epoch_callback=epoch_callback,
            device=torch.device("cpu"),
        )


def checkpoint_summary(model_path: Path) -> dict[str, object]:
    if not model_path.exists():
        return {"exists": False, "path": str(model_path)}
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        channel_index = checkpoint.get("channel_index")
        return {
            "exists": True,
            "path": str(model_path),
            "in_channels": int(checkpoint.get("in_channels", 1)),
            "channel_index": channel_index,
            "channel2_compatible": checkpoint.get("in_channels") == 1 and channel_index == CHANNEL_INDEX,
            "threshold": checkpoint.get("threshold", 0.5),
            "image_size": checkpoint.get("image_size"),
            "best_val_loss": checkpoint.get("best_val_loss"),
        }
    except Exception as exc:
        return {"exists": True, "path": str(model_path), "warning": str(exc)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/api/status")
def status() -> dict[str, object]:
    return {
        "project_root": str(ROOT),
        "output_root": str(OUTPUT_ROOT),
        "default_model": checkpoint_summary(DEFAULT_MODEL_PATH),
        "channel_index": CHANNEL_INDEX,
        "channel_note": "Zero-based CHANNEL_INDEX=1 means the second image channel only.",
        "image_size": IMAGE_SIZE,
        "device": str(get_device()),
        "supported_image_formats": sorted(SUPPORTED_IMAGE_EXTENSIONS),
    }


@app.get("/outputs/{path:path}")
def get_output(path: str):
    target = (OUTPUT_ROOT / path).resolve()
    try:
        target.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found.") from exc
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(target)


@app.post("/api/analyze-masks")
async def api_analyze_masks(
    mask_zip: UploadFile = File(...),
    min_object_size: int = Form(8),
):
    """Measure manually labeled cilia masks and create exploratory morphology t-SNE plots."""
    validate_zip_upload(mask_zip)
    if min_object_size < 1:
        raise HTTPException(status_code=400, detail="Minimum object size must be at least 1 pixel.")

    out_dir = ensure_output_dir("mask_tsne")
    with tempfile.TemporaryDirectory(prefix="cilia_app_mask_tsne_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "masks.zip"
        save_upload(mask_zip, zip_path)
        extract_dir = tmp_path / "masks"
        extract_dir.mkdir()
        safe_extract_zip(zip_path, extract_dir)
        mask_paths = find_mask_files(extract_dir)

        morphology_tables = []
        summaries = []
        invalid_masks = []
        for mask_path in mask_paths:
            try:
                mask = read_binary_mask_file(mask_path)
            except Exception as exc:
                invalid_masks.append(f"{mask_path.name}: {exc}")
                continue
            objects = measure_cilia_morphology(mask, min_area=min_object_size)
            relative_name = mask_path.relative_to(extract_dir).as_posix()
            summaries.append(
                {
                    "mask_file": relative_name,
                    "mask_pixels": int(mask.sum()),
                    "cilia_objects": int(len(objects)),
                }
            )
            if not objects.empty:
                morphology_tables.append(objects.assign(mask_file=relative_name, mask_stem=mask_path.stem))

    if invalid_masks:
        examples = "; ".join(invalid_masks[:3])
        suffix = "" if len(invalid_masks) <= 3 else f" (and {len(invalid_masks) - 3} more)"
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {len(invalid_masks)} uploaded mask file(s): {examples}{suffix}",
        )

    morphology_table = pd.concat(morphology_tables, ignore_index=True) if morphology_tables else pd.DataFrame()
    summary_path = out_dir / "mask_analysis_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    morphology_result = save_morphology_tsne(morphology_table, out_dir)
    return {
        "status": "ok",
        "message": "Mask morphology analysis completed. No model was run.",
        "analyzed_masks": len(summaries),
        "measured_cilia": int(len(morphology_table)),
        "minimum_object_size": min_object_size,
        "mask_summary_csv_url": relative_url(summary_path),
        "output_dir": str(out_dir),
        **morphology_result,
    }


@app.post("/api/predict")
async def api_predict(
    images: list[UploadFile] | None = File(None),
    image: UploadFile | None = File(None),
    model_path: str | None = Form(None),
    threshold: float = Form(0.5),
    full_resolution: bool = Form(True),
    tile_size: int = Form(512),
    overlap: int = Form(64),
    min_object_size: int = Form(8),
):
    uploads = [upload for upload in (images or []) if upload.filename]
    if image is not None and image.filename:
        uploads.append(image)
    if not uploads:
        raise HTTPException(status_code=400, detail="Select at least one supported multichannel image file.")
    suffixes = [validate_image_upload(upload) for upload in uploads]
    if not (0 < threshold < 1):
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1, for example 0.5.")
    if tile_size < 64:
        raise HTTPException(status_code=400, detail="tile_size should be at least 64 pixels.")
    if overlap < 0 or overlap >= tile_size:
        raise HTTPException(status_code=400, detail="overlap must be at least 0 and smaller than tile_size.")
    model_file = resolve_project_path(model_path, DEFAULT_MODEL_PATH)
    if not model_file.exists():
        raise HTTPException(status_code=400, detail=f"Model file does not exist: {model_file}")

    out_dir = ensure_output_dir("predict")
    masks_dir = out_dir / "masks"
    previews_dir = out_dir / "previews"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="cilia_app_predict_") as tmp:
            tmp_path = Path(tmp)
            model, checkpoint = load_unet_model(model_file)
            image_size = tuple(checkpoint.get("image_size") or IMAGE_SIZE)
            morphology_tables: list[pd.DataFrame] = []
            manifest_rows: list[dict[str, object]] = []
            prediction_items: list[dict[str, object]] = []
            used_stems: set[str] = set()

            for upload_index, (upload, suffix) in enumerate(zip(uploads, suffixes), start=1):
                original_name = Path(upload.filename or f"image_{upload_index}{suffix}").name
                upload_path = tmp_path / f"{upload_index:03d}_{original_name}"
                save_upload(upload, upload_path)
                stacks = prediction_stacks_from_file(upload_path)
                for image_index, stack in enumerate(stacks, start=1):
                    output_stem = prediction_output_stem(original_name, image_index, len(stacks), used_stems)
                    try:
                        if full_resolution and suffix in {".tif", ".tiff"}:
                            _, mask, channel2_hwc, objects = full_resolution_prediction_from_stack(
                                model,
                                stack,
                                threshold=threshold,
                                tile_size=tile_size,
                                overlap=overlap,
                                min_object_size=min_object_size,
                                source_name=output_stem,
                            )
                        else:
                            channel2_hwc = preprocess_image_array(stack, image_size=image_size, channel_index=CHANNEL_INDEX)
                            _, mask = predict_preprocessed(model, channel2_hwc, threshold=threshold)
                            objects = measure_cilia_morphology(mask, min_area=min_object_size)
                    except Exception as exc:
                        raise ValueError(f"{original_name}, image {image_index}: {exc}") from exc

                    mask_tif = masks_dir / f"{output_stem}_predicted_cilia_mask.tif"
                    mask_png = previews_dir / f"{output_stem}_predicted_cilia_mask.png"
                    overlay_png = previews_dir / f"{output_stem}_channel2_predicted_overlay.png"
                    write_mask(mask_tif, mask)
                    mask_png_file(mask, mask_png)
                    overlay_file_from_channel(channel2_hwc, mask, overlay_png)

                    mask_file = mask_tif.relative_to(out_dir).as_posix()
                    manifest_rows.append(
                        {
                            "source_image_file": original_name,
                            "source_image_stem": Path(original_name).stem,
                            "source_image_index": image_index,
                            "predicted_mask_file": mask_file,
                            "predicted_mask_stem": mask_tif.stem,
                            "cilia_objects": int(len(objects)),
                        }
                    )
                    if not objects.empty:
                        morphology_tables.append(
                            objects.assign(
                                mask_file=mask_file,
                                mask_stem=mask_tif.stem,
                                source_image_file=original_name,
                                source_image_stem=Path(original_name).stem,
                                source_image_index=image_index,
                            )
                        )
                    prediction_items.append(
                        {
                            "source_image_file": original_name,
                            "source_image_index": image_index,
                            "predicted_mask_tif_url": relative_url(mask_tif),
                            "mask_png_url": relative_url(mask_png),
                            "overlay_png_url": relative_url(overlay_png),
                            "object_count": int(len(objects)),
                        }
                    )
    except Exception as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=(
                "Prediction failed. Use supported multichannel images with at least two channels. "
                f"Only zero-based channel index {CHANNEL_INDEX} is used. Details: {exc}"
            ),
        ) from exc

    morphology_table = pd.concat(morphology_tables, ignore_index=True) if morphology_tables else pd.DataFrame()
    manifest_path = out_dir / "predicted_mask_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    mask_zip = out_dir / "predicted_cilia_masks.zip"
    with zipfile.ZipFile(mask_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in manifest_rows:
            mask_path = out_dir / str(item["predicted_mask_file"])
            archive.write(mask_path, arcname=str(item["predicted_mask_file"]))
        archive.write(manifest_path, arcname=manifest_path.name)

    morphology_result = save_morphology_tsne(morphology_table, out_dir)
    first_item = prediction_items[0]
    return {
        "status": "ok",
        "message": f"Predicted cilia masks for {len(prediction_items)} image(s) using only Channel 2 Input.",
        "processed_images": len(prediction_items),
        "predicted_masks_zip_url": relative_url(mask_zip),
        "mask_manifest_csv_url": relative_url(manifest_path),
        "prediction_items": prediction_items,
        "mask_tif_url": first_item["predicted_mask_tif_url"],
        "mask_png_url": first_item["mask_png_url"],
        "overlay_png_url": first_item["overlay_png_url"],
        "objects_csv_url": morphology_result["morphology_csv_url"],
        "object_count": int(len(morphology_table)),
        "output_dir": str(out_dir),
        **morphology_result,
    }


@app.post("/api/evaluate")
async def api_evaluate(
    dataset_zip: UploadFile = File(...),
    model_path: str | None = Form(None),
    threshold: float = Form(0.5),
    tile_size: int = Form(512),
    overlap: int = Form(64),
    min_object_size: int = Form(8),
    object_iou_threshold: float = Form(0.10),
):
    validate_zip_upload(dataset_zip)
    if not (0 < threshold < 1):
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1.")
    if tile_size < 64:
        raise HTTPException(status_code=400, detail="tile_size should be at least 64 pixels.")
    if overlap < 0 or overlap >= tile_size:
        raise HTTPException(status_code=400, detail="overlap must be at least 0 and smaller than tile_size.")
    if not (0 < object_iou_threshold <= 1):
        raise HTTPException(status_code=400, detail="object_iou_threshold must be greater than 0 and at most 1.")
    model_file = resolve_project_path(model_path, DEFAULT_MODEL_PATH)
    if not model_file.exists():
        raise HTTPException(status_code=400, detail=f"Model file does not exist: {model_file}")

    out_dir = ensure_output_dir("evaluate")
    with tempfile.TemporaryDirectory(prefix="cilia_app_eval_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "dataset.zip"
        save_upload(dataset_zip, zip_path)
        extract_dir = tmp_path / "dataset"
        extract_dir.mkdir()
        safe_extract_zip(zip_path, extract_dir)
        image_dir, mask_dir = find_training_dirs(extract_dir)
        try:
            pairs = find_image_mask_pairs(image_dir, mask_dir)
            model, _ = load_unet_model(model_file)
            rows = []
            morphology_tables = []
            roc_labels = []
            roc_scores = []
            pr_labels = []
            pr_scores = []
            pr_weights = []
            failure_records = []
            threshold_values = [round(float(value), 2) for value in np.arange(0.10, 0.91, 0.05)]
            sweep_totals = {value: {"intersection": 0.0, "predicted_pixels": 0.0, "true_pixels": 0.0, "union": 0.0} for value in threshold_values}
            object_totals = {"object_true_positive": 0, "object_false_positive": 0, "object_false_negative": 0}
            rng = np.random.default_rng(42)
            for pair in pairs:
                if pair.image_path.suffix.lower() not in {".tif", ".tiff"}:
                    raise ValueError("Full-resolution evaluation currently expects TIFF images.")
                true_mask = read_binary_mask_file(pair.mask_path)
                probability, pred_mask, channel2_hwc, objects = full_resolution_prediction(
                    model,
                    pair.image_path,
                    threshold=threshold,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_object_size=min_object_size,
                )
                if pred_mask.shape != true_mask.shape:
                    raise ValueError(f"{pair.stem}: predicted mask shape {pred_mask.shape} does not match true mask shape {true_mask.shape}.")
                metrics = binary_metrics(pred_mask, true_mask)
                object_metrics = object_detection_metrics(
                    pred_mask,
                    true_mask,
                    iou_threshold=object_iou_threshold,
                    min_object_size=min_object_size,
                )
                metrics.update(object_metrics)
                metrics["stem"] = pair.stem
                rows.append(metrics)
                for key in object_totals:
                    object_totals[key] += int(object_metrics[key])
                if not objects.empty:
                    morphology_tables.append(objects.assign(image_stem=pair.stem))
                labels, scores = sample_pixels_for_roc(probability, true_mask, rng)
                if len(labels):
                    roc_labels.append(labels)
                    roc_scores.append(scores)
                labels, scores, weights = sample_pixels_for_precision_recall(probability, true_mask, rng)
                if len(labels):
                    pr_labels.append(labels)
                    pr_scores.append(scores)
                    pr_weights.append(weights)
                for sweep_threshold in threshold_values:
                    sweep_mask = mask_from_probability(probability, sweep_threshold, min_object_size)
                    sweep_metrics = binary_metrics(sweep_mask, true_mask)
                    totals = sweep_totals[sweep_threshold]
                    totals["predicted_pixels"] += sweep_metrics["predicted_pixels"]
                    totals["true_pixels"] += sweep_metrics["true_pixels"]
                    totals["intersection"] += float(np.logical_and(sweep_mask.astype(bool), true_mask.astype(bool)).sum())
                    totals["union"] += float(np.logical_or(sweep_mask.astype(bool), true_mask.astype(bool)).sum())
                failure_records.append(
                    {
                        "stem": pair.stem,
                        "dice": metrics["dice"],
                        "channel2": channel2_hwc,
                        "true_mask": true_mask,
                        "pred_mask": pred_mask,
                    }
                )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Full-resolution evaluation failed. Check paired TIFF images and masks with matching stems. "
                    f"Details: {exc}"
                ),
            ) from exc

    table = pd.DataFrame(rows)
    metrics_path = out_dir / "test_metrics.csv"
    table.to_csv(metrics_path, index=False)
    mean_metrics = table[["dice", "iou", "predicted_pixels", "true_pixels"]].mean().to_dict()
    object_precision = object_totals["object_true_positive"] / max(object_totals["object_true_positive"] + object_totals["object_false_positive"], 1)
    object_recall = object_totals["object_true_positive"] / max(object_totals["object_true_positive"] + object_totals["object_false_negative"], 1)
    object_f1 = 2 * object_precision * object_recall / max(object_precision + object_recall, 1e-12)
    object_summary = {
        **object_totals,
        "object_precision": object_precision,
        "object_recall": object_recall,
        "object_f1": object_f1,
        "object_iou_threshold": object_iou_threshold,
    }
    sweep_rows = []
    for sweep_threshold, totals in sweep_totals.items():
        eps = 1e-6
        sweep_rows.append(
            {
                "threshold": sweep_threshold,
                "dice": (2 * totals["intersection"] + eps) / (totals["predicted_pixels"] + totals["true_pixels"] + eps),
                "iou": (totals["intersection"] + eps) / (totals["union"] + eps),
                "predicted_pixels": totals["predicted_pixels"],
                "true_pixels": totals["true_pixels"],
            }
        )
    morphology_table = pd.concat(morphology_tables, ignore_index=True) if morphology_tables else pd.DataFrame()
    morphology_result = save_morphology_tsne(morphology_table, out_dir)
    labels = np.concatenate(roc_labels) if roc_labels else np.array([], dtype=np.uint8)
    scores = np.concatenate(roc_scores) if roc_scores else np.array([], dtype=np.float32)
    roc_result = save_roc_auc_plot(labels, scores, out_dir)
    labels = np.concatenate(pr_labels) if pr_labels else np.array([], dtype=np.uint8)
    scores = np.concatenate(pr_scores) if pr_scores else np.array([], dtype=np.float32)
    weights = np.concatenate(pr_weights) if pr_weights else np.array([], dtype=np.float64)
    pr_result = save_precision_recall_plot(labels, scores, weights, out_dir)
    threshold_result = save_threshold_sweep_plot(sweep_rows, threshold, out_dir)
    failure_result = save_failure_review_gallery(failure_records, out_dir)
    return {
        "status": "ok",
        "tested_pairs": len(rows),
        "mean_metrics": mean_metrics,
        "object_metrics": object_summary,
        "metrics_csv_url": relative_url(metrics_path),
        "output_dir": str(out_dir),
        **morphology_result,
        **roc_result,
        **pr_result,
        **threshold_result,
        **failure_result,
    }


def job_error_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc)


def optimization_profile_args(profile: str) -> tuple[list[str], str]:
    if profile == "clean_baseline":
        return ["--background-ratio", "1.5", "--loss", "bce_dice"], "Clean baseline: source-image validation and enough background patches."
    if profile == "false_positive_focus":
        return [
            "--background-ratio", "2.5",
            "--loss", "bce_tversky",
            "--tversky-alpha", "0.70",
            "--tversky-beta", "0.30",
            "--rot90",
            "--intensity",
        ], "False-positive focus: more hard-negative patches and a false-positive-weighted Tversky loss."
    raise HTTPException(status_code=400, detail="Choose either the clean baseline or false-positive focus profile.")


def run_optimization_job(
    job_id: str,
    work_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    profile: str,
) -> None:
    log_tail: list[str] = []
    experiment_dir: Path | None = None
    try:
        profile_args, profile_note = optimization_profile_args(profile)
        update_training_job(job_id, status="queued", phase="Waiting for the training device", message="The optimization job is queued.")
        with TRAINING_RUN_LOCK:
            update_training_job(job_id, status="running", phase="Reading training ZIP", message="Checking paired images and masks before creating a source-image split.", progress_percent=2)
            extract_dir = work_dir / "dataset"
            extract_dir.mkdir()
            safe_extract_zip(work_dir / "dataset.zip", extract_dir)
            image_dir, mask_dir = find_training_dirs(extract_dir)
            pairs = find_image_mask_pairs(image_dir, mask_dir)
            if len(pairs) < 4:
                raise ValueError("Optimization needs at least four paired source images so whole images can be held out for validation.")

            OPTIMIZATION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            script_path = ROOT / "scripts" / "optimize_cilia.py"
            command = [
                sys.executable,
                str(script_path),
                "run",
                "--images", str(image_dir),
                "--masks", str(mask_dir),
                "--name", f"app_{profile}",
                "--output-root", str(OPTIMIZATION_OUTPUT_ROOT),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--learning-rate", str(learning_rate),
                "--patch-size", "512",
                "--stride", "256",
                "--min-positive-pixels", "15",
                "--validation-images", "3",
                *profile_args,
            ]
            environment = dict(os.environ)
            environment["MPLCONFIGDIR"] = str(work_dir / "matplotlib")
            Path(environment["MPLCONFIGDIR"]).mkdir(exist_ok=True)
            update_training_job(job_id, phase="Creating source-image split", message=profile_note, progress_percent=5)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                message = line.strip()
                if not message:
                    continue
                log_tail = (log_tail + [message])[-12:]
                if message.startswith("experiment="):
                    experiment_id = message.split("=", 1)[1].splitlines()[0]
                    experiment_dir = OPTIMIZATION_OUTPUT_ROOT / experiment_id
                    update_training_job(job_id, output_dir=str(experiment_dir))
                elif message.startswith("patches "):
                    update_training_job(job_id, phase="Preparing source-disjoint patches", message=message, progress_percent=12)
                else:
                    batch_match = re.search(r"epoch=(\d+)/(\d+) training_batch=(\d+)/(\d+)", message)
                    epoch_match = re.search(r"epoch=(\d+)/(\d+) train_loss=([0-9.]+) val_loss=([0-9.]+) val_dice=([0-9.]+)", message)
                    if batch_match:
                        epoch, total_epochs, batch, total_batches = (int(value) for value in batch_match.groups())
                        progress = 15 + round(65 * ((epoch - 1) + batch / max(total_batches, 1)) / max(total_epochs, 1))
                        update_training_job(job_id, phase=f"Training epoch {epoch} of {total_epochs}", message=f"Working through batch {batch} of {total_batches}. This can take several minutes on a laptop.", progress_percent=progress, completed_epochs=epoch - 1)
                    elif epoch_match:
                        epoch, total_epochs = int(epoch_match.group(1)), int(epoch_match.group(2))
                        metrics = {"train_loss": float(epoch_match.group(3)), "val_loss": float(epoch_match.group(4)), "val_dice": float(epoch_match.group(5))}
                        update_training_job(job_id, phase=f"Completed epoch {epoch} of {total_epochs}", message=f"Train loss {metrics['train_loss']:.4f}; validation Dice {metrics['val_dice']:.4f}.", progress_percent=15 + round(65 * epoch / max(total_epochs, 1)), completed_epochs=epoch, last_metrics=metrics)
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError("Optimization stopped. " + "\n".join(log_tail))
            if experiment_dir is None:
                candidates = sorted(OPTIMIZATION_OUTPUT_ROOT.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
                if candidates:
                    experiment_dir = candidates[-1].parent
            if experiment_dir is None or not (experiment_dir / "summary.json").exists():
                raise RuntimeError("Optimization finished without a readable summary file.")
            summary = json.loads((experiment_dir / "summary.json").read_text())
            figure_dir = experiment_dir / "figures"
            result = {
                "status": "ok",
                "message": "Optimization completed. The threshold was selected from held-out source images, not the test set.",
                "profile": profile,
                "experiment_id": summary.get("experiment_id"),
                "checkpoint": summary.get("checkpoint"),
                "validation": summary.get("validation", {}),
                "output_dir": str(experiment_dir),
                "training_curves_url": relative_url(figure_dir / "training_curves.png"),
                "validation_threshold_plot_url": relative_url(figure_dir / "validation_threshold_and_pr.png"),
                "validation_roc_plot_url": relative_url(figure_dir / "validation_roc.png"),
                "validation_false_positive_plot_url": relative_url(figure_dir / "validation_false_positive_analysis.png"),
            }
            update_training_job(job_id, status="completed", phase="Completed", message="Optimization results, false-positive analysis, and a new checkpoint are ready.", progress_percent=100, completed_epochs=epochs, result=result, finished_at=time.time())
    except Exception as exc:
        update_training_job(job_id, status="failed", phase="Optimization stopped", message="No existing checkpoint was changed.", error=job_error_message(exc), log_tail=log_tail, finished_at=time.time())
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_paired_training_job(
    job_id: str,
    work_dir: Path,
    out_dir: Path,
    target_model: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    patch_size: int,
    stride: int,
    min_positive_pixels: int,
    background_ratio: float,
    overwrite_model: bool,
) -> None:
    try:
        update_training_job(job_id, status="queued", phase="Waiting for the training device", message="Queued for the local training device.")
        with TRAINING_RUN_LOCK:
            update_training_job(job_id, status="running", phase="Reading training ZIP", message="Checking paired images and masks.", progress_percent=2)
            extract_dir = work_dir / "dataset"
            extract_dir.mkdir()
            safe_extract_zip(work_dir / "dataset.zip", extract_dir)
            image_dir, mask_dir = find_training_dirs(extract_dir)

            def patch_progress(completed: int, total: int) -> None:
                update_training_job(
                    job_id,
                    phase=f"Creating Channel 2 patches ({completed}/{total} images)",
                    message="Extracting only Channel 2 and matching it with the cilia masks.",
                    progress_percent=5 + round(25 * completed / max(total, 1)),
                )

            dataset_result = create_paired_patch_dataset(
                image_dir=image_dir,
                mask_dir=mask_dir,
                out_dir=out_dir,
                patch_size=patch_size,
                stride=stride,
                min_positive_pixels=min_positive_pixels,
                background_ratio=background_ratio,
                progress_callback=patch_progress,
            )
            if dataset_result["patch_count"] < 2:
                raise ValueError("Patch creation produced fewer than 2 patches. Try a smaller patch size, smaller stride, or lower min_positive_pixels.")

            def epoch_progress(completed: int, total: int, metrics: dict[str, float] | None) -> None:
                if metrics is None:
                    update_training_job(job_id, phase="Preparing patch stream", message="Checking one-channel patches without loading the full dataset into memory.", progress_percent=30)
                    return
                update_training_job(
                    job_id,
                    phase=f"Training epoch {completed} of {total}",
                    message=(
                        f"Train loss {metrics['train_loss']:.4f}; validation Dice {metrics['val_dice']:.4f}; "
                        f"validation IoU {metrics['val_iou']:.4f}."
                    ),
                    progress_percent=30 + round(65 * completed / max(total, 1)),
                    completed_epochs=completed,
                    last_metrics=metrics,
                )

            history = train_with_device_recovery(
                Path(dataset_result["patch_dir"]),
                model_path=target_model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_fraction=val_fraction,
                overwrite_model=overwrite_model,
                epoch_callback=epoch_progress,
                recovery_callback=lambda: update_training_job(
                    job_id,
                    phase="Recovering from Apple GPU error",
                    message="The Apple GPU closed its training pipe before a checkpoint was saved. Retrying once on CPU with the same data and settings.",
                    progress_percent=30,
                ),
            )
            update_training_job(job_id, phase="Saving model and plots", message="Writing the checkpoint, loss plot, and validation metrics.", progress_percent=97)
            plot_result = save_training_plots(history, out_dir)
            result = {
                "status": "ok",
                "message": "Patch-based training completed from paired images and masks using only Channel 2 Input.",
                "training_dataset": dataset_result,
                "saved_model_path": str(target_model),
                "final_metrics": history[-1] if history else {},
                "history": history,
                **plot_result,
            }
            update_training_job(job_id, status="completed", phase="Completed", message="Training finished. Your model and plots are ready.", progress_percent=100, completed_epochs=epochs, result=result, finished_at=time.time())
    except Exception as exc:
        update_training_job(job_id, status="failed", phase="Training stopped", message="Training could not be completed.", error=job_error_message(exc), finished_at=time.time())
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/api/training-jobs/{job_id}")
def api_training_job(job_id: str) -> dict[str, object]:
    return training_job_snapshot(job_id)


@app.post("/api/train-paired")
async def api_train_paired(
    dataset_zip: UploadFile = File(...),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    learning_rate: float = Form(1e-4),
    val_fraction: float = Form(0.2),
    patch_size: int = Form(512),
    stride: int = Form(256),
    min_positive_pixels: int = Form(3),
    background_ratio: float = Form(1.0),
    save_model_path: str | None = Form(None),
    overwrite_model: bool = Form(False),
    confirm_train: bool = Form(False),
):
    validate_training_params(epochs, batch_size, learning_rate, val_fraction, confirm_train)
    validate_zip_upload(dataset_zip)
    validate_patch_params(patch_size, stride, min_positive_pixels, background_ratio)
    out_dir = ensure_output_dir("paired_train")
    target_model = resolve_project_path(
        save_model_path,
        CHECKPOINT_ROOT / f"cilia_app_paired_{timestamp()}.pt",
    )
    if target_model.exists() and not overwrite_model:
        raise HTTPException(status_code=400, detail=f"Refusing to overwrite {target_model}. Enable overwrite or choose a new path.")

    work_dir = Path(tempfile.mkdtemp(prefix="cilia_app_train_paired_"))
    try:
        save_upload(dataset_zip, work_dir / "dataset.zip")
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    job_id = create_training_job("paired_masks", epochs)
    threading.Thread(
        target=run_paired_training_job,
        kwargs={
            "job_id": job_id,
            "work_dir": work_dir,
            "out_dir": out_dir,
            "target_model": target_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_fraction": val_fraction,
            "patch_size": patch_size,
            "stride": stride,
            "min_positive_pixels": min_positive_pixels,
            "background_ratio": background_ratio,
            "overwrite_model": overwrite_model,
        },
        daemon=True,
    ).start()
    return {
        "status": "accepted",
        "message": "Training started. This page will show preparation and epoch progress until the model and plots are ready.",
        "job_id": job_id,
        "status_url": f"/api/training-jobs/{job_id}",
    }


@app.post("/api/optimize")
async def api_optimize(
    dataset_zip: UploadFile = File(...),
    profile: str = Form("clean_baseline"),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    learning_rate: float = Form(1e-4),
    confirm_train: bool = Form(False),
):
    validate_zip_upload(dataset_zip)
    if not confirm_train:
        raise HTTPException(status_code=400, detail="Optimization was not started. Check confirm training before submitting.")
    if epochs <= 0:
        raise HTTPException(status_code=400, detail="epochs must be a positive integer.")
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be a positive integer.")
    if learning_rate <= 0:
        raise HTTPException(status_code=400, detail="learning_rate must be positive.")
    optimization_profile_args(profile)
    work_dir = Path(tempfile.mkdtemp(prefix="cilia_app_optimize_"))
    try:
        save_upload(dataset_zip, work_dir / "dataset.zip")
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    job_id = create_training_job("optimization", epochs)
    threading.Thread(
        target=run_optimization_job,
        kwargs={
            "job_id": job_id,
            "work_dir": work_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "profile": profile,
        },
        daemon=True,
    ).start()
    return {
        "status": "accepted",
        "message": "Optimization started. Existing checkpoints will not be overwritten, and the app will select the threshold only on held-out source images.",
        "job_id": job_id,
        "status_url": f"/api/training-jobs/{job_id}",
    }


def validate_training_params(epochs: int, batch_size: int, learning_rate: float, val_fraction: float, confirm_train: bool) -> None:
    if not confirm_train:
        raise HTTPException(status_code=400, detail="Training was not started. Check confirm_train before submitting.")
    if epochs <= 0:
        raise HTTPException(status_code=400, detail="epochs must be a positive integer.")
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be a positive integer.")
    if learning_rate <= 0:
        raise HTTPException(status_code=400, detail="learning_rate must be positive.")
    if not (0 <= val_fraction < 1):
        raise HTTPException(status_code=400, detail="val_fraction must be at least 0 and less than 1.")


def validate_patch_params(patch_size: int, stride: int, min_positive_pixels: int, background_ratio: float) -> None:
    if patch_size < 64:
        raise HTTPException(status_code=400, detail="patch_size should be at least 64 pixels.")
    if stride <= 0:
        raise HTTPException(status_code=400, detail="stride must be positive.")
    if stride > patch_size:
        raise HTTPException(status_code=400, detail="stride should be less than or equal to patch_size so patches cover the image.")
    if min_positive_pixels < 1:
        raise HTTPException(status_code=400, detail="min_positive_pixels must be at least 1.")
    if background_ratio < 0:
        raise HTTPException(status_code=400, detail="background_ratio cannot be negative.")


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cilia Detector</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --ink: #172026;
      --muted: #5b6872;
      --line: #d9e0e7;
      --panel: #ffffff;
      --accent: #166b8f;
      --accent-2: #376c3c;
      --warn: #9a5b10;
      --bad: #a33333;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }
    header p {
      margin: 0;
      color: var(--muted);
      max-width: 980px;
      line-height: 1.45;
    }
    nav {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 24px;
      border-bottom: 1px solid var(--line);
      background: #eef3f6;
    }
    nav button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 8px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
    }
    nav button.active {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    main {
      padding: 20px 24px 36px;
      max-width: 1220px;
      margin: 0 auto;
    }
    section {
      display: none;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
    }
    section.active { display: block; }
    h2 {
      margin: 0 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }
    h3 {
      margin: 18px 0 8px;
      font-size: 16px;
      letter-spacing: 0;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 12px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }
    input[type="checkbox"] {
      width: auto;
      margin-right: 7px;
    }
    .check {
      display: flex;
      align-items: center;
      min-height: 38px;
      color: var(--ink);
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    .primary, .secondary {
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    .primary { background: var(--accent); color: #fff; }
    .secondary { background: #e5ecef; color: var(--ink); }
    .note {
      border-left: 4px solid var(--accent-2);
      padding: 9px 12px;
      margin: 12px 0;
      background: #eef7ef;
      color: #26362a;
      line-height: 1.45;
    }
    .warn {
      border-left-color: var(--warn);
      background: #fff7e8;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #111820;
      color: #e8f0f2;
      border-radius: 6px;
      padding: 12px;
      overflow: auto;
      min-height: 80px;
    }
    .preview {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }
    .job-progress {
      display: none;
      margin-top: 14px;
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      padding: 11px 12px;
      background: #f3f8fa;
    }
    .job-progress strong { display: block; margin-bottom: 4px; }
    .job-progress span { color: var(--muted); font-size: 14px; }
    .job-progress progress { width: 100%; height: 14px; margin-top: 10px; accent-color: var(--accent); }
    figure {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }
    figure img {
      display: block;
      width: 100%;
      height: auto;
      background: #000;
    }
    figcaption {
      padding: 9px 10px;
      font-weight: 700;
      color: var(--muted);
      font-size: 13px;
    }
    a { color: var(--accent); font-weight: 700; }
    code {
      background: #edf2f4;
      padding: 1px 4px;
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Cilia Detector</h1>
    <p>Train, test, predict, and inspect cilia segmentation. The U-Net input is always the second image channel only: zero-based <code>CHANNEL_INDEX = 1</code>.</p>
  </header>
  <nav>
    <button class="active" data-tab="status">Status</button>
    <button data-tab="predict">Predict</button>
    <button data-tab="evaluate">Test</button>
    <button data-tab="analyze-masks">Analyze Masks</button>
    <button data-tab="train-paired">Train From Masks</button>
  </nav>
  <main>
    <section id="status" class="active">
      <h2>Status</h2>
      <div class="note">Prediction needs only new multichannel images. Training needs matching image and manually prepared cilia-mask files.</div>
      <div class="actions">
        <button class="primary" onclick="loadStatus()">Refresh Status</button>
      </div>
      <pre id="statusOutput"></pre>
    </section>

    <section id="predict">
      <h2>Predict Cilia</h2>
      <div class="note">Select one or more multichannel images. TIFF time or scene stacks are processed as separate images automatically. The app saves one predicted mask per image, creates collective cilia t-SNE plots, and provides downloads for the masks and coordinates.</div>
      <form id="predictForm">
        <div class="grid">
          <label>Image file(s)<input name="images" type="file" accept=".tif,.tiff,.npy,.npz,.png,.jpg,.jpeg" multiple required></label>
          <label>Model path<input name="model_path" placeholder="checkpoints/channel2_uploaded_cilia_unet.pt"></label>
          <label>Threshold<input name="threshold" type="number" min="0.01" max="0.99" step="0.01" value="0.5"></label>
          <label>Tile size<input name="tile_size" type="number" min="64" step="32" value="512"></label>
          <label>Overlap<input name="overlap" type="number" min="0" step="16" value="64"></label>
          <label>Minimum object size<input name="min_object_size" type="number" min="1" step="1" value="8"></label>
          <label class="check"><input name="full_resolution" type="checkbox" checked> Full-resolution TIFF prediction</label>
        </div>
        <div class="actions">
          <button class="primary" type="submit">Run Prediction</button>
        </div>
      </form>
      <pre id="predictOutput"></pre>
      <div id="predictSummary" class="note" style="display:none"></div>
      <div id="predictPreview" class="preview"></div>
    </section>

    <section id="evaluate">
      <h2>Test A Model</h2>
      <div class="note">Upload a ZIP containing paired test images and ground-truth masks with matching filenames. Choose a trained model and probability threshold, then run the test to view segmentation scores, diagnostic plots, and failure reviews.</div>
      <form id="evaluateForm">
        <div class="grid">
          <label>Dataset ZIP<input name="dataset_zip" type="file" accept=".zip" required></label>
          <label>Model path<input name="model_path" placeholder="checkpoints/channel2_uploaded_cilia_unet.pt"></label>
          <label>Threshold<input name="threshold" type="number" min="0.01" max="0.99" step="0.01" value="0.5"></label>
          <label>Tile size<input name="tile_size" type="number" min="64" step="32" value="512"></label>
          <label>Overlap<input name="overlap" type="number" min="0" step="16" value="64"></label>
          <label>Minimum object size<input name="min_object_size" type="number" min="1" step="1" value="8"></label>
          <label>Object match IoU<input name="object_iou_threshold" type="number" min="0.01" max="1" step="0.01" value="0.10"></label>
        </div>
        <div class="actions">
          <button class="primary" type="submit">Run Test</button>
        </div>
      </form>
      <pre id="evaluateOutput"></pre>
      <div id="evaluateSummary" class="note" style="display:none"></div>
      <div id="evaluatePreview" class="preview"></div>
    </section>

    <section id="analyze-masks">
      <h2>Analyze Cilia Masks</h2>
      <div class="note">Upload a ZIP of binary cilia masks to explore cilia size, length, and curvature. Each mask should have cilia as nonzero pixels and background as zero. Choose the smallest object size to include, then select Analyze Masks.</div>
      <form id="analyzeMasksForm">
        <div class="grid">
          <label>Mask ZIP<input name="mask_zip" type="file" accept=".zip" required></label>
          <label>Minimum object size<input name="min_object_size" type="number" min="1" step="1" value="8" required></label>
        </div>
        <div class="actions">
          <button class="primary" type="submit">Create t-SNE Plots</button>
        </div>
      </form>
      <div id="maskAnalysisSummary" class="note" style="display:none"></div>
      <pre id="analyzeMasksOutput"></pre>
      <div id="analyzeMasksPreview" class="preview"></div>
    </section>

    <section id="train-paired">
      <h2>Train From Image And Mask Folders</h2>
      <div class="note warn">Upload a ZIP containing multi-channel images and matching binary cilia masks. Each image and mask must have the same filename stem. Choose training settings, check Confirm training, then select Train Model. Training uses full-resolution patches and shows progress while it runs.</div>
      <form id="trainPairedForm">
        <div class="grid">
          <label>Dataset ZIP<input name="dataset_zip" type="file" accept=".zip" required></label>
          <label>Save model path<input name="save_model_path" placeholder="checkpoints/my_cilia_model.pt"></label>
          <label>Epochs<input name="epochs" type="number" min="1" value="10" required></label>
          <label>Batch size<input name="batch_size" type="number" min="1" value="4" required></label>
          <label>Learning rate<input name="learning_rate" type="number" min="0.000001" step="0.000001" value="0.0001"></label>
          <label>Validation fraction<input name="val_fraction" type="number" min="0" max="0.9" step="0.05" value="0.2"></label>
          <label>Patch size<input name="patch_size" type="number" min="64" step="32" value="512"></label>
          <label>Stride<input name="stride" type="number" min="1" step="16" value="256"></label>
          <label>Minimum positive pixels<input name="min_positive_pixels" type="number" min="1" step="1" value="3"></label>
          <label>Background patches per positive patch<input name="background_ratio" type="number" min="0" step="0.25" value="1.0"></label>
          <label class="check"><input name="confirm_train" type="checkbox"> Confirm training</label>
          <label class="check"><input name="overwrite_model" type="checkbox"> Allow overwrite</label>
        </div>
        <div class="actions">
          <button class="primary" type="submit">Train Model</button>
        </div>
      </form>
      <div id="trainPairedProgress" class="job-progress" aria-live="polite">
        <strong id="trainPairedPhase">Preparing training</strong>
        <span id="trainPairedMessage"></span>
        <progress id="trainPairedBar" value="0" max="100"></progress>
      </div>
      <pre id="trainPairedOutput"></pre>
      <div id="trainPairedPreview" class="preview"></div>
    </section>

  </main>
  <script>
    const tabs = document.querySelectorAll("nav button");
    const sections = document.querySelectorAll("main section");
    tabs.forEach(button => {
      button.addEventListener("click", () => {
        tabs.forEach(b => b.classList.remove("active"));
        sections.forEach(s => s.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab).classList.add("active");
      });
    });

    function formData(form) {
      const data = new FormData(form);
      form.querySelectorAll('input[type="checkbox"]').forEach(input => {
        data.set(input.name, input.checked ? "true" : "false");
      });
      return data;
    }

    async function postForm(url, form, outputId) {
      const output = document.getElementById(outputId);
      output.textContent = "Working...";
      const response = await fetch(url, { method: "POST", body: formData(form) });
      const text = await response.text();
      let payload;
      try { payload = JSON.parse(text); } catch { payload = { raw: text }; }
      if (!response.ok) {
        output.textContent = JSON.stringify(payload, null, 2);
        throw new Error(payload.detail || "Request failed");
      }
      output.textContent = JSON.stringify(payload, null, 2);
      return payload;
    }

    async function loadStatus() {
      const output = document.getElementById("statusOutput");
      output.textContent = "Loading...";
      const response = await fetch("/api/status");
      output.textContent = JSON.stringify(await response.json(), null, 2);
    }

    function imageFigure(url, label) {
      return `<figure><img src="${url}" alt="${label}"><figcaption>${label} <a href="${url}" target="_blank">open</a></figcaption></figure>`;
    }

    function resultFigures(payload, items) {
      return items.filter(([key]) => payload[key]).map(([key, label]) => imageFigure(payload[key], label)).join("");
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>'"]/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
      })[char]);
    }

    function showEvaluationSummary(payload) {
      const mean = payload.mean_metrics || {};
      const objects = payload.object_metrics || {};
      const pieces = [
        `Pixel Dice: ${(mean.dice ?? 0).toFixed(3)}`,
        `Pixel IoU: ${(mean.iou ?? 0).toFixed(3)}`,
        `Pixel AP: ${(payload.pixel_average_precision ?? 0).toFixed(3)}`,
        `Object F1: ${(objects.object_f1 ?? 0).toFixed(3)}`,
        `Best Dice threshold: ${(payload.best_dice_threshold ?? 0).toFixed(2)}`,
      ];
      const summary = document.getElementById("evaluateSummary");
      summary.textContent = pieces.join(" | ");
      summary.style.display = "block";
    }

    function setTrainingProgress(prefix, job) {
      document.getElementById(`${prefix}Progress`).style.display = "block";
      document.getElementById(`${prefix}Phase`).textContent = `${job.phase} (${job.progress_percent}%)`;
      document.getElementById(`${prefix}Message`).textContent = `${job.message} Elapsed: ${job.elapsed_seconds}s.`;
      document.getElementById(`${prefix}Bar`).value = job.progress_percent;
    }

    function sleep(milliseconds) {
      return new Promise(resolve => setTimeout(resolve, milliseconds));
    }

    async function startTraining(url, form, outputId, prefix, previewId) {
      const output = document.getElementById(outputId);
      const submit = form.querySelector('button[type="submit"]');
      output.textContent = "Starting training...";
      submit.disabled = true;
      try {
        const response = await fetch(url, { method: "POST", body: formData(form) });
        const payload = await response.json();
        if (!response.ok) {
          output.textContent = JSON.stringify(payload, null, 2);
          throw new Error(payload.detail || "Training could not be started.");
        }
        while (true) {
          const jobResponse = await fetch(payload.status_url);
          const job = await jobResponse.json();
          if (!jobResponse.ok) throw new Error(job.detail || "Could not read training progress.");
          setTrainingProgress(prefix, job);
          output.textContent = JSON.stringify(job, null, 2);
          if (job.status === "completed") {
            const result = job.result || {};
            const figures = [
              ["loss_plot_url", "Training and Validation Loss"],
              ["validation_metrics_plot_url", "Validation Dice and IoU"],
            ];
            document.getElementById(previewId).innerHTML = resultFigures(result, figures);
            return result;
          }
          if (job.status === "failed") throw new Error(job.error || "Training stopped.");
          await sleep(1000);
        }
      } catch (error) {
        output.textContent += `\n\nError: ${error.message}`;
      } finally {
        submit.disabled = false;
      }
    }

    document.getElementById("predictForm").addEventListener("submit", async event => {
      event.preventDefault();
      const payload = await postForm("/api/predict", event.target, "predictOutput");
      const items = payload.prediction_items || [];
      const first = items[0] || payload;
      const links = [
        payload.predicted_masks_zip_url ? `<a href="${payload.predicted_masks_zip_url}" target="_blank">predicted masks ZIP</a>` : "",
        payload.mask_manifest_csv_url ? `<a href="${payload.mask_manifest_csv_url}" target="_blank">mask manifest CSV</a>` : "",
        payload.tsne_coordinates_csv_url ? `<a href="${payload.tsne_coordinates_csv_url}" target="_blank">t-SNE coordinates CSV</a>` : "",
      ].filter(Boolean).join(" | ");
      const individualMasks = items.map(item =>
        `<a href="${item.predicted_mask_tif_url}" target="_blank">${escapeHtml(item.source_image_file)} image ${item.source_image_index} mask</a>`
      ).join(" | ");
      const summary = document.getElementById("predictSummary");
      summary.innerHTML = `<strong>${(payload.processed_images ?? items.length ?? 1)} image(s) processed; ${payload.object_count ?? 0} cilium object(s) measured.</strong> ${escapeHtml(payload.tsne_note || "")} ${links}${individualMasks ? `<br>${individualMasks}` : ""}`;
      summary.style.display = "block";
      document.getElementById("predictPreview").innerHTML = [
        first.overlay_png_url ? imageFigure(first.overlay_png_url, "Channel 2 Input + Predicted Cilia Mask") : "",
        first.mask_png_url ? imageFigure(first.mask_png_url, "Predicted Cilia Mask") : "",
        resultFigures(payload, [
          ["tsne_length_plot_url", "t-SNE: Cilia Length"],
          ["tsne_curvature_plot_url", "t-SNE: Cilia Curvature"],
        ]),
      ].join("");
    });

    document.getElementById("evaluateForm").addEventListener("submit", async event => {
      event.preventDefault();
      const payload = await postForm("/api/evaluate", event.target, "evaluateOutput");
      showEvaluationSummary(payload);
      document.getElementById("evaluatePreview").innerHTML = resultFigures(payload, [
        ["threshold_sweep_plot_url", "Threshold Sweep: Dice and IoU"],
        ["roc_plot_url", "Pixel ROC/AUC"],
        ["pr_plot_url", "Pixel Precision-Recall and Average Precision"],
        ["tsne_length_plot_url", "t-SNE: Predicted Cilia Length"],
        ["tsne_curvature_plot_url", "t-SNE: Predicted Cilia Curvature"],
      ]) + (payload.failure_review || []).map(item =>
        imageFigure(item.url, `Failure Review: ${item.stem} (Dice ${item.dice.toFixed(3)})`)
      ).join("");
    });

    document.getElementById("analyzeMasksForm").addEventListener("submit", async event => {
      event.preventDefault();
      const payload = await postForm("/api/analyze-masks", event.target, "analyzeMasksOutput");
      const links = [
        payload.mask_summary_csv_url ? `<a href="${payload.mask_summary_csv_url}" target="_blank">mask summary CSV</a>` : "",
        payload.morphology_csv_url ? `<a href="${payload.morphology_csv_url}" target="_blank">cilia measurements CSV</a>` : "",
        payload.tsne_coordinates_csv_url ? `<a href="${payload.tsne_coordinates_csv_url}" target="_blank">t-SNE coordinates CSV</a>` : "",
      ].filter(Boolean).join(" | ");
      const summary = document.getElementById("maskAnalysisSummary");
      summary.innerHTML = `<strong>${payload.analyzed_masks} mask(s), ${payload.measured_cilia} measured cilium object(s).</strong> ${escapeHtml(payload.tsne_note || "")} ${links}`;
      summary.style.display = "block";
      document.getElementById("analyzeMasksPreview").innerHTML = resultFigures(payload, [
        ["tsne_length_plot_url", "t-SNE: Cilia Length"],
        ["tsne_curvature_plot_url", "t-SNE: Cilia Curvature"],
      ]);
    });

    document.getElementById("trainPairedForm").addEventListener("submit", async event => {
      event.preventDefault();
      await startTraining("/api/train-paired", event.target, "trainPairedOutput", "trainPaired", "trainPairedPreview");
    });

    loadStatus();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
