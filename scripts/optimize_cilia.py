#!/usr/bin/env python3
"""Reproducible, false-positive-focused experiments for the Channel 2 cilia U-Net.

This script deliberately does not modify the browser-app workflow or any existing
checkpoint. It uses an image-level train/validation split before patches are made,
then writes a self-contained experiment directory with configuration, split,
checkpoint, metrics, object analysis, and figures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from skimage import measure, morphology
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cilia_detector.channel2_pipeline import CHANNEL_INDEX, PairRecord, find_image_mask_pairs, load_unet_model, read_array
from cilia_detector.denoise import denoise_single_channel_input
from cilia_detector.io import read_stack
from cilia_detector.model import UNet, dice_loss
from cilia_detector.morphology import measure_cilia_morphology


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "optimization_20260910"
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.10, 0.96, 0.05))
MASK_COLORMAP = ListedColormap(["#ffffff", "#d62728"])
FALSE_POSITIVE_COLOR = "#0072b2"
FALSE_NEGATIVE_COLOR = "#7b2cbf"
TRUE_POSITIVE_COLOR = "#2e7d32"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int = 42
    patch_size: int = 512
    stride: int = 256
    min_positive_pixels: int = 15
    background_ratio: float = 1.5
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    bce_pos_weight: float = 6.0
    loss_name: str = "bce_dice"
    tversky_alpha: float = 0.70
    tversky_beta: float = 0.30
    augment_rot90: bool = False
    augment_intensity: bool = False
    patience: int = 8
    min_object_size: int = 8
    object_iou_threshold: float = 0.10
    validation_images: int = 3


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def read_mask(path: Path) -> np.ndarray:
    mask = np.squeeze(np.asarray(read_array(path)))
    if mask.ndim == 3:
        if mask.shape[0] <= 4:
            mask = mask[0]
        elif mask.shape[-1] <= 4:
            mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"{path.name} must be a 2D mask; received {mask.shape}.")
    return (mask > 0).astype(np.uint8)


def load_channel2(path: Path) -> np.ndarray:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Full-resolution optimization expects TIFF images, not {path.name}.")
    stack, _ = read_stack(path)
    channel = denoise_single_channel_input(stack, channel_index=CHANNEL_INDEX)
    if channel.ndim != 3 or channel.shape[0] != 1:
        raise ValueError(f"{path.name} did not produce a second-channel-only input; received {channel.shape}.")
    return channel


def split_by_source(pairs: list[PairRecord], validation_images: int, seed: int) -> tuple[list[PairRecord], list[PairRecord]]:
    if len(pairs) < 4:
        raise ValueError("At least four paired source images are required for an image-level train/validation split.")
    if validation_images < 1 or validation_images >= len(pairs):
        raise ValueError("validation_images must be between 1 and one fewer than the number of paired images.")
    order = np.random.default_rng(seed).permutation(len(pairs))
    validation_indexes = set(int(index) for index in order[:validation_images])
    training = [pair for index, pair in enumerate(pairs) if index not in validation_indexes]
    validation = [pair for index, pair in enumerate(pairs) if index in validation_indexes]
    return training, validation


def patch_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    positions = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions


def _extract_patch(image: np.ndarray, mask: np.ndarray, y: int, x: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    return image[:, y : y + size, x : x + size].astype(np.float32), mask[y : y + size, x : x + size].astype(np.float32)


def make_source_patches(pair: PairRecord, config: ExperimentConfig, rng: np.random.Generator) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    image = load_channel2(pair.image_path)
    mask = read_mask(pair.mask_path)
    if image.shape[-2:] != mask.shape:
        raise ValueError(f"{pair.stem}: Channel 2 is {image.shape[-2:]}, but mask is {mask.shape}.")
    positive: list[tuple[np.ndarray, np.ndarray]] = []
    background: list[tuple[np.ndarray, np.ndarray]] = []
    seen_coordinates: set[tuple[int, int]] = set()
    for y in patch_positions(mask.shape[0], config.patch_size, config.stride):
        for x in patch_positions(mask.shape[1], config.patch_size, config.stride):
            seen_coordinates.add((y, x))
            item = _extract_patch(image, mask, y, x, config.patch_size)
            if int(item[1].sum()) >= config.min_positive_pixels:
                positive.append(item)
            else:
                background.append(item)

    requested_background = int(np.ceil(len(positive) * config.background_ratio))
    # Grid tiles alone are too few to honour a negative ratio on 2048px images.
    # Add random, all-background training patches until the configured ratio is met.
    attempts = 0
    max_attempts = max(500, requested_background * 80)
    while len(background) < requested_background and attempts < max_attempts:
        y = int(rng.integers(0, mask.shape[0] - config.patch_size + 1))
        x = int(rng.integers(0, mask.shape[1] - config.patch_size + 1))
        attempts += 1
        if (y, x) in seen_coordinates:
            continue
        seen_coordinates.add((y, x))
        item = _extract_patch(image, mask, y, x, config.patch_size)
        if int(item[1].sum()) < config.min_positive_pixels:
            background.append(item)
    rng.shuffle(background)
    selected = positive + background[:requested_background]
    rng.shuffle(selected)
    return selected, {
        "positive_patches": len(positive),
        "background_patches_selected": min(len(background), requested_background),
        "background_patches_available": len(background),
        "background_patches_requested": requested_background,
    }


class PatchDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray]], config: ExperimentConfig, augment: bool):
        for image, _ in samples:
            if image.ndim != 3 or image.shape[0] != 1:
                raise ValueError(
                    f"Optimization expects second-channel-only patches shaped (1, height, width), got {image.shape}."
                )
        self.samples = samples
        self.config = config
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.samples[index]
        image, mask = image.copy(), mask.copy()
        if self.augment:
            if random.random() < 0.5:
                image, mask = np.flip(image, axis=2).copy(), np.flip(mask, axis=1).copy()
            if random.random() < 0.5:
                image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=0).copy()
            if self.config.augment_rot90:
                turns = random.randrange(4)
                image, mask = np.rot90(image, turns, axes=(1, 2)).copy(), np.rot90(mask, turns).copy()
            if self.config.augment_intensity:
                scale = random.uniform(0.92, 1.08)
                shift = random.uniform(-0.025, 0.025)
                image = np.clip(image * scale + shift, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(image).float(), torch.from_numpy(mask[None, ...]).float()


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float, beta: float, focal_gamma: float | None = None) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, targets.ndim))
    tp = torch.sum(probs * targets, dims)
    fp = torch.sum(probs * (1 - targets), dims)
    fn = torch.sum((1 - probs) * targets, dims)
    score = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    loss = 1 - score
    return (loss.pow(focal_gamma) if focal_gamma is not None else loss).mean()


def loss_function(logits: torch.Tensor, targets: torch.Tensor, bce: torch.nn.Module, config: ExperimentConfig) -> torch.Tensor:
    if config.loss_name == "bce_dice":
        return bce(logits, targets) + dice_loss(logits, targets)
    if config.loss_name == "bce_tversky":
        return bce(logits, targets) + tversky_loss(logits, targets, config.tversky_alpha, config.tversky_beta)
    if config.loss_name == "focal_tversky":
        return tversky_loss(logits, targets, config.tversky_alpha, config.tversky_beta, focal_gamma=1.33)
    raise ValueError(f"Unknown loss_name {config.loss_name!r}.")


def patch_metric_totals(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int, int]:
    pred = torch.sigmoid(logits) >= 0.5
    true = targets.bool()
    return (
        int(torch.logical_and(pred, true).sum().item()),
        int(torch.logical_and(pred, ~true).sum().item()),
        int(torch.logical_and(~pred, true).sum().item()),
        int(torch.logical_and(~pred, ~true).sum().item()),
    )


def metrics_from_counts(tp: int | float, fp: int | float, fn: int | float, tn: int | float) -> dict[str, float]:
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fp / (fp + tn + eps),
        "fnr": fn / (fn + tp + eps),
        "dice": dice,
        "iou": iou,
        "tp_pixels": float(tp),
        "fp_pixels": float(fp),
        "fn_pixels": float(fn),
        "tn_pixels": float(tn),
    }


def train_model(train_samples: list[tuple[np.ndarray, np.ndarray]], val_samples: list[tuple[np.ndarray, np.ndarray]], config: ExperimentConfig, checkpoint: Path) -> tuple[list[dict[str, float]], int]:
    work_device = device()
    model = UNet(in_channels=1).to(work_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.bce_pos_weight], device=work_device))
    train_loader = DataLoader(PatchDataset(train_samples, config, augment=True), batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(PatchDataset(val_samples, config, augment=False), batch_size=config.batch_size, shuffle=False, num_workers=0)
    history: list[dict[str, float]] = []
    best_loss, best_epoch, stalled = float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss, train_seen = 0.0, 0
        for batch_index, (images, masks) in enumerate(train_loader, start=1):
            images, masks = images.to(work_device), masks.to(work_device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), masks, bce, config)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * images.size(0)
            train_seen += images.size(0)
            if batch_index % max(1, len(train_loader) // 4) == 0 or batch_index == len(train_loader):
                print(f"epoch={epoch:03d}/{config.epochs:03d} training_batch={batch_index}/{len(train_loader)}", flush=True)
        model.eval()
        val_loss, val_seen = 0.0, 0
        counts = np.zeros(4, dtype=np.int64)
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(work_device), masks.to(work_device)
                logits = model(images)
                val_loss += float(loss_function(logits, masks, bce, config).item()) * images.size(0)
                val_seen += images.size(0)
                counts += patch_metric_totals(logits, masks)
        train_loss /= max(train_seen, 1)
        val_loss /= max(val_seen, 1)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"], **{f"val_{key}": value for key, value in metrics_from_counts(*counts).items()}}
        history.append(row)
        print(f"epoch={epoch:03d}/{config.epochs:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={row['val_dice']:.4f} lr={row['learning_rate']:.2g} seconds={time.perf_counter() - epoch_start:.1f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stalled = val_loss, epoch, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "in_channels": 1, "channel_index": CHANNEL_INDEX, "threshold": 0.5, "best_val_loss": best_loss, "experiment_config": asdict(config), "best_epoch": best_epoch}, checkpoint)
        else:
            stalled += 1
            if stalled >= config.patience:
                print(f"early_stop: no validation-loss improvement for {config.patience} epochs", flush=True)
                break
    return history, best_epoch


def predict_tiled(model: UNet, image: np.ndarray, tile_size: int = 512, overlap: int = 64) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(
            f"Optimization prediction expects one second-channel input shaped (1, height, width), got {image.shape}."
        )
    work_device = device()
    _, height, width = image.shape
    starts_y = patch_positions(height, tile_size, tile_size - overlap)
    starts_x = patch_positions(width, tile_size, tile_size - overlap)
    probability = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    model = model.to(work_device).eval()
    with torch.no_grad():
        for y in starts_y:
            for x in starts_x:
                tile = torch.from_numpy(image[:, y : y + tile_size, x : x + tile_size][None]).float().to(work_device)
                pred = torch.sigmoid(model(tile))[0, 0].detach().cpu().numpy()
                probability[y : y + tile_size, x : x + tile_size] += pred
                weights[y : y + tile_size, x : x + tile_size] += 1
    return probability / np.maximum(weights, 1e-6)


def clean_prediction(probability: np.ndarray, threshold: float, min_object_size: int) -> np.ndarray:
    return morphology.remove_small_objects(probability >= threshold, min_size=min_object_size).astype(np.uint8)


def match_objects(pred_mask: np.ndarray, true_mask: np.ndarray, iou_threshold: float) -> tuple[dict[str, int], set[int], set[int], np.ndarray, np.ndarray]:
    pred_labels = measure.label(pred_mask.astype(bool))
    true_labels = measure.label(true_mask.astype(bool))
    pred_count, true_count = int(pred_labels.max()), int(true_labels.max())
    candidates: list[tuple[float, int, int]] = []
    if pred_count and true_count:
        overlap = (pred_labels > 0) & (true_labels > 0)
        pred_area = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)
        true_area = np.bincount(true_labels.ravel(), minlength=true_count + 1)
        codes, intersections = np.unique(true_labels[overlap] * (pred_count + 1) + pred_labels[overlap], return_counts=True)
        for code, intersection in zip(codes, intersections):
            true_id, pred_id = int(code // (pred_count + 1)), int(code % (pred_count + 1))
            score = float(intersection / (true_area[true_id] + pred_area[pred_id] - intersection))
            if score >= iou_threshold:
                candidates.append((score, true_id, pred_id))
    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    for _, true_id, pred_id in sorted(candidates, reverse=True):
        if true_id not in matched_true and pred_id not in matched_pred:
            matched_true.add(true_id)
            matched_pred.add(pred_id)
    return {"object_tp": len(matched_pred), "object_fp": pred_count - len(matched_pred), "object_fn": true_count - len(matched_true), "object_predicted_count": pred_count, "object_true_count": true_count}, matched_pred, matched_true, pred_labels, true_labels


def object_table(stem: str, labels: np.ndarray, probability: np.ndarray, channel: np.ndarray, matched: set[int], kind: str) -> pd.DataFrame:
    table = measure_cilia_morphology(labels > 0, min_area=1)
    if table.empty:
        return pd.DataFrame(columns=["stem", "kind", "matched", "mean_probability", "mean_channel2_intensity", "max_probability", "distance_to_border_px"])
    rows = []
    height, width = labels.shape
    for region in measure.regionprops(labels):
        object_id = int(region.label)
        region_mask = labels == object_id
        measurement = table.loc[table["object_id"] == object_id]
        values = measurement.iloc[0].to_dict() if not measurement.empty else {"object_id": object_id, "area_px": int(region.area)}
        y, x = region.centroid
        rows.append({
            **values, "stem": stem, "kind": kind, "matched": object_id in matched,
            "mean_probability": float(probability[region_mask].mean()), "max_probability": float(probability[region_mask].max()),
            "mean_channel2_intensity": float(channel[region_mask].mean()),
            "distance_to_border_px": float(min(y, x, height - 1 - y, width - 1 - x)),
        })
    return pd.DataFrame(rows)


def save_case_figure(stem: str, channel: np.ndarray, prediction: np.ndarray, truth: np.ndarray, path: Path) -> None:
    """Save a source-image review with red masks and object error callouts."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(channel, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")
    axes[1].imshow(truth, cmap=MASK_COLORMAP, vmin=0, vmax=1)
    axes[1].set_title("True Cilia Mask")
    axes[2].imshow(prediction, cmap=MASK_COLORMAP, vmin=0, vmax=1)
    axes[2].set_title("Predicted Cilia Mask")
    circles = object_review_circles(prediction, truth)
    draw_object_circles(axes[2], circles, "true_positive")
    draw_object_circles(axes[2], circles, "false_positive")
    draw_object_circles(axes[2], circles, "false_negative")
    axes[0].axis("off")
    for axis in axes[1:]:
        style_mask_axis(axis)
    fig.suptitle(stem)
    add_object_legend(fig, circles)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def object_review_circles(prediction: np.ndarray, truth: np.ndarray, iou_threshold: float = 0.10) -> list[dict[str, float | str]]:
    """Locate matched and unmatched cilia for reviewer callout circles."""
    _, matched_pred, matched_true, pred_labels, true_labels = match_objects(prediction, truth, iou_threshold)
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
        min_row, min_col, max_row, max_col = region.bbox
        circles.append({
            "kind": "false_negative",
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
        if item["kind"] == kind:
            axis.add_patch(Circle((float(item["col"]), float(item["row"])), radius=float(item["radius"]), fill=False, edgecolor=color, linewidth=1.2, linestyle="-"))


def style_mask_axis(axis: plt.Axes) -> None:
    """Keep a thin black frame around white mask panels."""
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


def save_training_plot(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss", color="#166b8f")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss", color="#b35a2a")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss")
    axes[1].plot(history["epoch"], history["val_dice"], label="Validation Dice", color="#376c3c")
    axes[1].plot(history["epoch"], history["val_iou"], label="Validation IoU", color="#704c9b")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation Patch Metrics", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_model(
    model_path: Path,
    pairs: list[PairRecord],
    config: ExperimentConfig,
    out_dir: Path,
    label: str,
    selected_threshold: float | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model, _ = load_unet_model(model_path, device=device())
    probabilities: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    channels: list[np.ndarray] = []
    stems: list[str] = []
    for pair in pairs:
        image = load_channel2(pair.image_path)
        truth = read_mask(pair.mask_path)
        probabilities.append(predict_tiled(model, image))
        truths.append(truth)
        channels.append(image[0])
        stems.append(pair.stem)

    threshold_rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        counts = np.zeros(4, dtype=np.int64)
        object_counts = np.zeros(5, dtype=np.int64)
        for probability, truth in zip(probabilities, truths):
            pred = clean_prediction(probability, threshold, config.min_object_size)
            counts += [np.logical_and(pred, truth).sum(), np.logical_and(pred, ~truth.astype(bool)).sum(), np.logical_and(~pred.astype(bool), truth).sum(), np.logical_and(~pred.astype(bool), ~truth.astype(bool)).sum()]
            obj, *_ = match_objects(pred, truth, config.object_iou_threshold)
            object_counts += [obj["object_tp"], obj["object_fp"], obj["object_fn"], obj["object_predicted_count"], obj["object_true_count"]]
        object_precision = object_counts[0] / max(object_counts[0] + object_counts[1], 1)
        object_recall = object_counts[0] / max(object_counts[0] + object_counts[2], 1)
        threshold_rows.append({"threshold": threshold, **metrics_from_counts(*counts), "object_precision": object_precision, "object_recall": object_recall, "object_f1": 2 * object_precision * object_recall / max(object_precision + object_recall, 1e-9), "object_tp": int(object_counts[0]), "object_fp": int(object_counts[1]), "object_fn": int(object_counts[2]), "object_predicted_count": int(object_counts[3]), "object_true_count": int(object_counts[4])})
    sweep = pd.DataFrame(threshold_rows)
    best = sweep.loc[sweep["dice"].idxmax()]
    threshold = float(best["threshold"]) if selected_threshold is None else float(selected_threshold)
    selected = sweep.iloc[(sweep["threshold"] - threshold).abs().argsort().iloc[0]]
    metrics_rows: list[dict[str, object]] = []
    object_rows: list[pd.DataFrame] = []
    gallery = out_dir / f"{label}_cases"
    gallery.mkdir(exist_ok=True)
    for stem, channel, probability, truth in zip(stems, channels, probabilities, truths):
        pred = clean_prediction(probability, threshold, config.min_object_size)
        pixel = metrics_from_counts(np.logical_and(pred, truth).sum(), np.logical_and(pred, ~truth.astype(bool)).sum(), np.logical_and(~pred.astype(bool), truth).sum(), np.logical_and(~pred.astype(bool), ~truth.astype(bool)).sum())
        obj, matched_pred, matched_true, pred_labels, true_labels = match_objects(pred, truth, config.object_iou_threshold)
        metrics_rows.append({"stem": stem, **pixel, **obj, "predicted_pixels": int(pred.sum()), "true_pixels": int(truth.sum())})
        object_rows.extend([object_table(stem, pred_labels, probability, channel, matched_pred, "predicted"), object_table(stem, true_labels, probability, channel, matched_true, "ground_truth")])
        save_case_figure(stem, channel, pred, truth, gallery / f"{stem}_review.png")
    per_image = pd.DataFrame(metrics_rows)
    objects = pd.concat(object_rows, ignore_index=True) if object_rows else pd.DataFrame()
    flat_truth = np.concatenate([truth.ravel() for truth in truths])
    flat_probability = np.concatenate([prob.ravel() for prob in probabilities])
    summary = {"label": label, "source_images": len(pairs), "selected_threshold": threshold, "best_dice_threshold_diagnostic": float(best["threshold"]), **{key: float(selected[key]) for key in ["precision", "recall", "specificity", "fpr", "fnr", "dice", "iou", "object_precision", "object_recall", "object_f1"]}, **{key: int(selected[key]) for key in ["object_tp", "object_fp", "object_fn", "object_predicted_count", "object_true_count"]}, "pixel_roc_auc": float(roc_auc_score(flat_truth, flat_probability)), "pixel_average_precision": float(average_precision_score(flat_truth, flat_probability))}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(sweep["threshold"], sweep["precision"], marker="o", label="Precision")
    axes[0].plot(sweep["threshold"], sweep["recall"], marker="o", label="Recall")
    axes[0].plot(sweep["threshold"], sweep["dice"], marker="o", label="Dice")
    axes[0].plot(sweep["threshold"], sweep["fpr"], marker="o", label="False-positive rate")
    axes[0].axvline(threshold, color="#444", linestyle="--", label=f"Selected {threshold:.2f}")
    axes[0].set(xlabel="Threshold", ylabel="Score", title=f"{label.title()} Threshold Analysis", ylim=(0, 1))
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)
    precision, recall, _ = precision_recall_curve(flat_truth, flat_probability)
    axes[1].plot(recall, precision, color="#376c3c", label=f"PR (AP {summary['pixel_average_precision']:.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"{label.title()} Pixel Precision-Recall", xlim=(0, 1), ylim=(0, 1))
    axes[1].grid(alpha=0.25); axes[1].legend()
    fig.tight_layout(); fig.savefig(out_dir / f"{label}_threshold_and_pr.png", dpi=160); plt.close(fig)
    fpr, tpr, _ = roc_curve(flat_truth, flat_probability)
    fig, ax = plt.subplots(figsize=(5, 4.5)); ax.plot(fpr, tpr, label=f"AUC {summary['pixel_roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], "--", color="#777"); ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{label.title()} Pixel ROC/AUC", xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(out_dir / f"{label}_roc.png", dpi=160); plt.close(fig)
    return summary, per_image, sweep, objects


def plot_false_positive_analysis(objects: pd.DataFrame, path: Path) -> None:
    predicted = objects.loc[objects.get("kind", pd.Series(dtype=str)) == "predicted"].copy()
    if predicted.empty:
        return
    predicted["class"] = np.where(predicted["matched"], "Matched prediction", "False positive")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, column, title in zip(axes, ["area_px", "mean_channel2_intensity", "mean_probability"], ["Object area", "Channel 2 intensity", "Mean predicted probability"]):
        for name, group in predicted.groupby("class"):
            axis.hist(group[column].dropna(), bins=20, alpha=0.65, label=name)
        axis.set(title=title, xlabel=column, ylabel="Objects")
        axis.legend(fontsize=8); axis.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def run_experiment(args: argparse.Namespace) -> Path:
    config = ExperimentConfig(
        name=args.name, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        patch_size=args.patch_size, stride=args.stride, min_positive_pixels=args.min_positive_pixels,
        background_ratio=args.background_ratio, loss_name=args.loss, bce_pos_weight=args.bce_pos_weight,
        tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta, augment_rot90=args.rot90,
        augment_intensity=args.intensity, validation_images=args.validation_images, patience=args.patience,
        min_object_size=args.min_object_size,
    )
    set_seed(config.seed)
    experiment_id = f"{now_id()}_{config.name}"
    out_dir = Path(args.output_root) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "figures").mkdir()
    pairs = find_image_mask_pairs(args.images, args.masks)
    train_pairs, val_pairs = split_by_source(pairs, config.validation_images, config.seed)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (out_dir / "source_split.json").write_text(json.dumps({"training_stems": [pair.stem for pair in train_pairs], "validation_stems": [pair.stem for pair in val_pairs], "seed": config.seed}, indent=2))
    print(f"experiment={experiment_id}\ntrain_sources={[pair.stem for pair in train_pairs]}\nvalidation_sources={[pair.stem for pair in val_pairs]}", flush=True)
    rng = np.random.default_rng(config.seed)
    train_samples: list[tuple[np.ndarray, np.ndarray]] = []
    val_samples: list[tuple[np.ndarray, np.ndarray]] = []
    patch_rows = []
    for partition, records, destination in [("train", train_pairs, train_samples), ("validation", val_pairs, val_samples)]:
        for pair in records:
            samples, counts = make_source_patches(pair, config, rng)
            destination.extend(samples)
            patch_rows.append({"partition": partition, "stem": pair.stem, **counts, "selected_patches": len(samples)})
            print(f"patches {partition} {pair.stem}: {counts} selected={len(samples)}", flush=True)
    pd.DataFrame(patch_rows).to_csv(out_dir / "patch_sampling.csv", index=False)
    if not train_samples or not val_samples:
        raise ValueError("Patch creation produced no train or validation samples. Reduce minimum positive pixels or patch size.")
    history, best_epoch = train_model(train_samples, val_samples, config, out_dir / "best_model.pt")
    history_table = pd.DataFrame(history); history_table.to_csv(out_dir / "training_history.csv", index=False); save_training_plot(history_table, out_dir / "figures" / "training_curves.png")
    validation_summary, val_metrics, val_sweep, val_objects = evaluate_model(out_dir / "best_model.pt", val_pairs, config, out_dir / "figures", "validation")
    val_metrics.to_csv(out_dir / "validation_metrics.csv", index=False); val_sweep.to_csv(out_dir / "validation_thresholds.csv", index=False); val_objects.to_csv(out_dir / "validation_objects.csv", index=False); plot_false_positive_analysis(val_objects, out_dir / "figures" / "validation_false_positive_analysis.png")
    result: dict[str, object] = {"experiment_id": experiment_id, "device": str(device()), "best_epoch": best_epoch, "validation": validation_summary, "checkpoint": str(out_dir / "best_model.pt")}
    if args.test_images and args.test_masks:
        test_pairs = find_image_mask_pairs(args.test_images, args.test_masks)
        # Threshold is selected only from validation data; test evaluation applies it unchanged.
        test_summary, test_metrics, test_sweep, test_objects = evaluate_model(
            out_dir / "best_model.pt",
            test_pairs,
            config,
            out_dir / "figures",
            "test_exploratory",
            selected_threshold=float(validation_summary["selected_threshold"]),
        )
        test_summary["selected_threshold_from_validation"] = validation_summary["selected_threshold"]
        result["test_exploratory"] = test_summary
        test_metrics.to_csv(out_dir / "test_metrics.csv", index=False); test_sweep.to_csv(out_dir / "test_thresholds_diagnostic_only.csv", index=False); test_objects.to_csv(out_dir / "test_objects.csv", index=False); plot_false_positive_analysis(test_objects, out_dir / "figures" / "test_false_positive_analysis.png")
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return out_dir


def compare_experiments(args: argparse.Namespace) -> Path:
    result_dirs = [Path(value) for value in args.experiments]
    rows = []
    for result_dir in result_dirs:
        summary = json.loads((result_dir / "summary.json").read_text())
        for split in ("validation", "test_exploratory"):
            if split in summary:
                rows.append({"experiment_id": summary["experiment_id"], "split": split, **summary[split], "checkpoint": summary["checkpoint"]})
    table = pd.DataFrame(rows).sort_values(["split", "dice"], ascending=[True, False])
    output = Path(args.output_root) / f"comparison_{now_id()}"; output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8)); pivot = table.pivot(index="experiment_id", columns="split", values="object_fp"); pivot.plot(kind="bar", ax=ax, color=["#d73027", "#fdae61"]); ax.set(ylabel="False-positive objects", title="False-positive Object Comparison"); ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(output / "false_positive_comparison.png", dpi=160); plt.close(fig)
    return output


def evaluate_existing_checkpoint(args: argparse.Namespace) -> Path:
    config = ExperimentConfig(
        name=args.name,
        seed=args.seed,
        min_object_size=args.min_object_size,
        object_iou_threshold=args.object_iou_threshold,
    )
    out_dir = Path(args.output_root) / f"{now_id()}_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "figures").mkdir()
    pairs = find_image_mask_pairs(args.images, args.masks)
    summary, per_image, sweep, objects = evaluate_model(
        Path(args.checkpoint),
        pairs,
        config,
        out_dir / "figures",
        args.label,
        selected_threshold=args.threshold,
    )
    result = {
        "experiment_id": out_dir.name,
        "kind": "existing_checkpoint_diagnostic",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "note": "This is a diagnostic baseline. It does not train or alter the checkpoint, and its threshold is supplied explicitly rather than selected on this dataset.",
        "evaluation": summary,
    }
    per_image.to_csv(out_dir / "metrics.csv", index=False)
    sweep.to_csv(out_dir / "thresholds_diagnostic_only.csv", index=False)
    objects.to_csv(out_dir / "objects.csv", index=False)
    plot_false_positive_analysis(objects, out_dir / "figures" / "false_positive_analysis.png")
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return out_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Train one reproducible source-split experiment.")
    run.add_argument("--images", required=True, help="Folder of full-resolution multichannel TIFF images.")
    run.add_argument("--masks", required=True, help="Folder of matching 2D binary masks.")
    run.add_argument("--test-images", help="Optional held-out image folder. Results are reported only after validation selection.")
    run.add_argument("--test-masks", help="Optional held-out mask folder.")
    run.add_argument("--name", required=True, help="Short experiment label, such as baseline_source_split.")
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run.add_argument("--seed", type=int, default=42); run.add_argument("--validation-images", type=int, default=3)
    run.add_argument("--epochs", type=int, default=20); run.add_argument("--batch-size", type=int, default=4); run.add_argument("--learning-rate", type=float, default=1e-4); run.add_argument("--patience", type=int, default=8)
    run.add_argument("--patch-size", type=int, default=512); run.add_argument("--stride", type=int, default=256); run.add_argument("--min-positive-pixels", type=int, default=15); run.add_argument("--background-ratio", type=float, default=1.5)
    run.add_argument("--loss", choices=["bce_dice", "bce_tversky", "focal_tversky"], default="bce_dice"); run.add_argument("--bce-pos-weight", type=float, default=6.0); run.add_argument("--tversky-alpha", type=float, default=0.70); run.add_argument("--tversky-beta", type=float, default=0.30)
    run.add_argument("--rot90", action="store_true"); run.add_argument("--intensity", action="store_true"); run.add_argument("--min-object-size", type=int, default=8)
    compare = subparsers.add_parser("compare", help="Create a baseline-vs-candidate comparison from completed experiment directories.")
    compare.add_argument("experiments", nargs="+", help="Completed experiment directories containing summary.json.")
    compare.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    evaluate = subparsers.add_parser("evaluate", help="Run a read-only detailed diagnostic on an existing checkpoint.")
    evaluate.add_argument("--checkpoint", required=True, help="Existing .pt checkpoint; it is never modified.")
    evaluate.add_argument("--images", required=True, help="Folder of full-resolution TIFF images.")
    evaluate.add_argument("--masks", required=True, help="Folder of matching binary masks.")
    evaluate.add_argument("--name", required=True, help="Short diagnostic label.")
    evaluate.add_argument("--label", default="baseline", help="Figure label.")
    evaluate.add_argument("--threshold", type=float, required=True, help="Explicit fixed probability threshold, between 0 and 1.")
    evaluate.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--min-object-size", type=int, default=8)
    evaluate.add_argument("--object-iou-threshold", type=float, default=0.10)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "run":
        run_experiment(args)
    elif args.command == "compare":
        print(compare_experiments(args))
    else:
        if not 0 < args.threshold < 1:
            raise ValueError("--threshold must be between 0 and 1.")
        evaluate_existing_checkpoint(args)


if __name__ == "__main__":
    main()
