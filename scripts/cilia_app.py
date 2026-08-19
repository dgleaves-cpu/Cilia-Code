#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import random
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from skimage import morphology
from torch.utils.data import DataLoader, Dataset, random_split

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
    preprocess_image_file,
    read_array,
    select_channel_preserve_dim,
)
from cilia_detector.classify import classify_objects
from cilia_detector.dataset import iter_patches, make_training_arrays, save_patch_npz
from cilia_detector.denoise import denoise_single_channel_input, normalize_percentile
from cilia_detector.io import read_stack, rois_to_mask, extract_imagej_rois, write_mask
from cilia_detector.model import UNet, dice_loss


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "app"
CHECKPOINT_ROOT = ROOT / "checkpoints"
SUPPORTED_ZIP_EXTENSIONS = {".zip"}


def choose_default_model_path() -> Path:
    preferred = [
        ROOT / MODEL_PATH,
        CHECKPOINT_ROOT / "channel2_uploaded_cilia_unet.pt",
        CHECKPOINT_ROOT / "cilia_unet.pt",
    ]
    for path in preferred:
        if path.exists():
            return path
    checkpoints = sorted(CHECKPOINT_ROOT.glob("*.pt")) if CHECKPOINT_ROOT.exists() else []
    return checkpoints[0] if checkpoints else ROOT / MODEL_PATH


DEFAULT_MODEL_PATH = choose_default_model_path()

app = FastAPI(
    title="Cilia Detector App",
    description="Browser app for channel-2 cilia U-Net training, testing, prediction, ROI extraction, and object measurement.",
    version="0.2.0",
)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_output_dir(prefix: str) -> Path:
    out_dir = OUTPUT_ROOT / f"{prefix}_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


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

    for base in candidates:
        image_dir = child_dir_case_insensitive(base, "images")
        mask_dir = child_dir_case_insensitive(base, "masks")
        if image_dir is not None and mask_dir is not None:
            return image_dir, mask_dir
    raise HTTPException(
        status_code=400,
        detail="Could not find images/ and masks/ folders. Expected images/sample.tif and masks/sample.tif with matching stems.",
    )


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
    selected = denoise_single_channel_input(stack, channel_index=CHANNEL_INDEX)
    probability = predict_tiled(model, selected, get_device(), tile_size=tile_size, overlap=overlap)
    mask = probability >= threshold
    mask = morphology.remove_small_objects(mask, min_size=min_object_size).astype(np.uint8)
    channel2_hwc = np.moveaxis(selected, 0, -1)
    objects = classify_objects(stack, mask) if stack.shape[-2:] == mask.shape else pd.DataFrame()
    return probability, mask, channel2_hwc, objects


def resized_prediction(model: UNet, image_path: Path, image_size: tuple[int, int], threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel2 = preprocess_image_file(image_path, image_size=image_size, channel_index=CHANNEL_INDEX)
    probability, mask = predict_preprocessed(model, channel2, threshold=threshold)
    return probability, mask, channel2


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def predict_tiled(model: UNet, inputs: np.ndarray, device: torch.device, tile_size: int = 512, overlap: int = 64) -> np.ndarray:
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
        return denoise_single_channel_input(stack, channel_index=CHANNEL_INDEX)
    image = read_array(image_path)
    channel_hwc = select_channel_preserve_dim(image, channel_index=CHANNEL_INDEX)
    channel_hwc = normalize_selected_channel(channel_hwc)
    return np.moveaxis(channel_hwc, -1, 0).astype(np.float32)


def iter_training_patches_balanced(
    inputs: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    stride: int,
    min_positive_pixels: int,
    background_ratio: float,
    seed: int = 42,
):
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
    for pair in pairs:
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
    def __init__(self, patch_dir: Path, preload: bool = True, augment: bool = True):
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
                mask = torch.from_numpy(data["mask"][None, ...]).float()
                self.preloaded.append((image, mask))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        if self.preloaded is None:
            data = np.load(self.paths[idx])
            image = torch.from_numpy(data["image"]).float()
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
) -> list[dict[str, float]]:
    if model_path.exists() and not overwrite_model:
        raise HTTPException(status_code=400, detail=f"Refusing to overwrite {model_path}. Enable overwrite or choose a new path.")

    device = get_device()
    load_start = time.perf_counter()
    dataset = PatchDataset(patch_dir, preload=True, augment=True)
    print(f"Loaded {len(dataset)} patches into memory in {time.perf_counter() - load_start:.1f}s", flush=True)
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


def create_roi_patch_dataset(extract_dir: Path, out_dir: Path, patch_size: int, stride: int, label_channel: int, line_width: int) -> dict[str, object]:
    image_paths = sorted([p for p in extract_dir.rglob("*") if p.suffix.lower() in {".tif", ".tiff"}])
    if not image_paths:
        raise HTTPException(status_code=400, detail="No .tif or .tiff files were found in the uploaded ZIP.")

    rows = []
    total_patches = 0
    patch_dir = out_dir / "patches"
    roi_dir = out_dir / "roi_masks"
    patch_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        inputs, mask = make_training_arrays(
            image_path,
            label_channel=label_channel,
            channel_index=CHANNEL_INDEX,
            line_width=line_width,
        )
        mask_path = roi_dir / f"{image_path.stem}_roi_cilia_mask.tif"
        write_mask(mask_path, mask)

        patch_count = 0
        for image_patch, mask_patch in iter_patches(inputs, mask, patch_size=patch_size, stride=stride):
            save_patch_npz(patch_dir / f"{image_path.stem}_patch{patch_count:04d}.npz", image_patch, mask_patch, str(image_path))
            patch_count += 1
            total_patches += 1
        rows.append(
            {
                "image": image_path.name,
                "mask_pixels": int(mask.sum()),
                "patches": patch_count,
                "roi_mask": str(mask_path),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "roi_patch_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {
        "image_count": len(image_paths),
        "patch_count": total_patches,
        "patch_dir": str(patch_dir),
        "summary_path": str(summary_path),
        "summary_url": relative_url(summary_path),
        "images": rows,
    }


def checkpoint_summary(model_path: Path) -> dict[str, object]:
    if not model_path.exists():
        return {"exists": False, "path": str(model_path)}
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        return {
            "exists": True,
            "path": str(model_path),
            "in_channels": int(checkpoint.get("in_channels", 1)),
            "channel_index": checkpoint.get("channel_index", CHANNEL_INDEX),
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


@app.post("/api/predict")
async def api_predict(
    image: UploadFile = File(...),
    model_path: str | None = Form(None),
    threshold: float = Form(0.5),
    full_resolution: bool = Form(True),
    tile_size: int = Form(512),
    overlap: int = Form(64),
    min_object_size: int = Form(8),
):
    suffix = validate_image_upload(image)
    if not (0 < threshold < 1):
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1, for example 0.5.")
    model_file = resolve_project_path(model_path, DEFAULT_MODEL_PATH)
    if not model_file.exists():
        raise HTTPException(status_code=400, detail=f"Model file does not exist: {model_file}")

    out_dir = ensure_output_dir("predict")
    with tempfile.TemporaryDirectory(prefix="cilia_app_predict_") as tmp:
        upload_path = Path(tmp) / f"uploaded{suffix}"
        save_upload(image, upload_path)
        try:
            model, checkpoint = load_unet_model(model_file)
            if full_resolution and suffix in {".tif", ".tiff"}:
                _, mask, channel2_hwc, objects = full_resolution_prediction(
                    model,
                    upload_path,
                    threshold=threshold,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_object_size=min_object_size,
                )
            else:
                image_size = tuple(checkpoint.get("image_size") or IMAGE_SIZE)
                _, mask, channel2_hwc = resized_prediction(model, upload_path, image_size=image_size, threshold=threshold)
                objects = pd.DataFrame()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Prediction failed. Upload a supported multichannel image in HWC or CHW order with at least two "
                    f"channels. Only zero-based channel index {CHANNEL_INDEX} is used. Details: {exc}"
                ),
            ) from exc

    mask_tif = out_dir / "predicted_cilia_mask.tif"
    mask_png = out_dir / "predicted_cilia_mask.png"
    overlay_png = out_dir / "channel2_predicted_overlay.png"
    write_mask(mask_tif, mask)
    mask_png_file(mask, mask_png)
    overlay_file_from_channel(channel2_hwc, mask, overlay_png)

    objects_url = None
    objects_path = None
    object_count = None
    if not objects.empty:
        objects_path = out_dir / "cilia_objects.csv"
        objects.to_csv(objects_path, index=False)
        objects_url = relative_url(objects_path)
        object_count = int(len(objects))

    return {
        "status": "ok",
        "message": "Prediction completed using only Channel 2 Input.",
        "mask_tif_url": relative_url(mask_tif),
        "mask_png_url": relative_url(mask_png),
        "overlay_png_url": relative_url(overlay_png),
        "objects_csv_url": objects_url,
        "object_count": object_count,
        "output_dir": str(out_dir),
    }


@app.post("/api/evaluate")
async def api_evaluate(
    dataset_zip: UploadFile = File(...),
    model_path: str | None = Form(None),
    threshold: float = Form(0.5),
    tile_size: int = Form(512),
    overlap: int = Form(64),
    min_object_size: int = Form(8),
):
    validate_zip_upload(dataset_zip)
    if not (0 < threshold < 1):
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1.")
    if tile_size < 64:
        raise HTTPException(status_code=400, detail="tile_size should be at least 64 pixels.")
    if overlap < 0 or overlap >= tile_size:
        raise HTTPException(status_code=400, detail="overlap must be at least 0 and smaller than tile_size.")
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
            for pair in pairs:
                if pair.image_path.suffix.lower() not in {".tif", ".tiff"}:
                    raise ValueError("Full-resolution evaluation currently expects TIFF images.")
                true_mask = read_binary_mask_file(pair.mask_path)
                _, pred_mask, _, _ = full_resolution_prediction(
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
                metrics["stem"] = pair.stem
                rows.append(metrics)
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
    return {
        "status": "ok",
        "tested_pairs": len(rows),
        "mean_metrics": mean_metrics,
        "metrics_csv_url": relative_url(metrics_path),
        "output_dir": str(out_dir),
    }


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

    with tempfile.TemporaryDirectory(prefix="cilia_app_train_paired_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "dataset.zip"
        save_upload(dataset_zip, zip_path)
        extract_dir = tmp_path / "dataset"
        extract_dir.mkdir()
        safe_extract_zip(zip_path, extract_dir)
        image_dir, mask_dir = find_training_dirs(extract_dir)
        try:
            dataset_result = create_paired_patch_dataset(
                image_dir=image_dir,
                mask_dir=mask_dir,
                out_dir=out_dir,
                patch_size=patch_size,
                stride=stride,
                min_positive_pixels=min_positive_pixels,
                background_ratio=background_ratio,
            )
            if dataset_result["patch_count"] < 2:
                raise ValueError(
                    "Patch creation produced fewer than 2 patches. Try a smaller patch size, smaller stride, "
                    "or lower min_positive_pixels."
                )
            history = train_from_patch_dir(
                Path(dataset_result["patch_dir"]),
                model_path=target_model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_fraction=val_fraction,
                overwrite_model=overwrite_model,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Patch-based training failed. Check that images and masks have matching stems, that masks are "
                    f"binary and full resolution, and that Channel 2 exists. Details: {exc}"
                ),
            ) from exc

    return {
        "status": "ok",
        "message": "Patch-based training completed from paired images and masks using only Channel 2 Input.",
        "training_dataset": dataset_result,
        "saved_model_path": str(target_model),
        "final_metrics": history[-1] if history else {},
        "history": history,
    }


@app.post("/api/train-roi")
async def api_train_roi(
    image_zip: UploadFile = File(...),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    learning_rate: float = Form(1e-4),
    val_fraction: float = Form(0.2),
    patch_size: int = Form(256),
    stride: int = Form(128),
    label_channel: int = Form(2),
    line_width: int = Form(5),
    save_model_path: str | None = Form(None),
    overwrite_model: bool = Form(False),
    confirm_train: bool = Form(False),
):
    validate_training_params(epochs, batch_size, learning_rate, val_fraction, confirm_train)
    validate_zip_upload(image_zip)
    if patch_size < 64:
        raise HTTPException(status_code=400, detail="patch_size should be at least 64 pixels.")
    if stride <= 0:
        raise HTTPException(status_code=400, detail="stride must be positive.")

    out_dir = ensure_output_dir("roi_train")
    target_model = resolve_project_path(
        save_model_path,
        CHECKPOINT_ROOT / f"cilia_app_roi_{timestamp()}.pt",
    )

    with tempfile.TemporaryDirectory(prefix="cilia_app_train_roi_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "images.zip"
        save_upload(image_zip, zip_path)
        extract_dir = tmp_path / "images"
        extract_dir.mkdir()
        safe_extract_zip(zip_path, extract_dir)
        try:
            roi_result = create_roi_patch_dataset(
                extract_dir,
                out_dir=out_dir,
                patch_size=patch_size,
                stride=stride,
                label_channel=label_channel,
                line_width=line_width,
            )
            history = train_from_patch_dir(
                Path(roi_result["patch_dir"]),
                model_path=target_model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_fraction=val_fraction,
                overwrite_model=overwrite_model,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "ROI training failed. Upload analyzed ImageJ/FIJI TIFFs that contain overlay ROIs. "
                    f"Details: {exc}"
                ),
            ) from exc

    return {
        "status": "ok",
        "message": "ROI labels were extracted, patches were created, and training completed.",
        "saved_model_path": str(target_model),
        "roi_dataset": roi_result,
        "final_metrics": history[-1] if history else {},
        "history": history,
    }


@app.post("/api/extract-rois")
async def api_extract_rois(
    image_zip: UploadFile = File(...),
    label_channel: int = Form(2),
    line_width: int = Form(5),
):
    validate_zip_upload(image_zip)
    out_dir = ensure_output_dir("roi_extract")
    with tempfile.TemporaryDirectory(prefix="cilia_app_extract_roi_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "images.zip"
        save_upload(image_zip, zip_path)
        extract_dir = tmp_path / "images"
        extract_dir.mkdir()
        safe_extract_zip(zip_path, extract_dir)
        image_paths = sorted([p for p in extract_dir.rglob("*") if p.suffix.lower() in {".tif", ".tiff"}])
        if not image_paths:
            raise HTTPException(status_code=400, detail="No .tif or .tiff files were found in the uploaded ZIP.")
        rows = []
        for image_path in image_paths:
            stack, metadata = read_stack(image_path)
            rois = extract_imagej_rois(metadata)
            mask = rois_to_mask(rois, stack.shape[-2:], channel=label_channel, line_width=line_width)
            if mask.sum() == 0:
                mask = rois_to_mask(rois, stack.shape[-2:], channel=None, line_width=line_width)
            mask_path = out_dir / f"{image_path.stem}_roi_cilia_mask.tif"
            png_path = out_dir / f"{image_path.stem}_roi_overlay.png"
            write_mask(mask_path, mask)
            channel = normalize_percentile(stack[CHANNEL_INDEX])
            overlay_file_from_channel(channel[..., np.newaxis], mask, png_path)
            rows.append(
                {
                    "image": image_path.name,
                    "roi_count": len(rois),
                    "mask_pixels": int(mask.sum()),
                    "mask_url": relative_url(mask_path),
                    "overlay_url": relative_url(png_path),
                }
            )
    summary_path = out_dir / "roi_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    return {"status": "ok", "images": rows, "summary_url": relative_url(summary_path), "output_dir": str(out_dir)}


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
    <button data-tab="train-paired">Train From Masks</button>
    <button data-tab="train-roi">Train From ImageJ ROIs</button>
    <button data-tab="extract-rois">Extract ROI Masks</button>
  </nav>
  <main>
    <section id="status" class="active">
      <h2>Status</h2>
      <div class="note">Prediction needs only new multichannel images. Training needs labels, either matching mask files or ImageJ/FIJI ROIs embedded in analyzed TIFFs.</div>
      <div class="actions">
        <button class="primary" onclick="loadStatus()">Refresh Status</button>
      </div>
      <pre id="statusOutput"></pre>
    </section>

    <section id="predict">
      <h2>Predict Cilia</h2>
      <div class="note">Upload one multichannel image. TIFF files can be predicted at full resolution; other supported formats are resized to the model input size.</div>
      <form id="predictForm">
        <div class="grid">
          <label>Image file<input name="image" type="file" required></label>
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
      <div id="predictPreview" class="preview"></div>
    </section>

    <section id="evaluate">
      <h2>Test A Model</h2>
      <div class="note">Upload a ZIP with paired <code>images/</code> and <code>masks/</code> folders. Files are matched by filename stem.</div>
      <form id="evaluateForm">
        <div class="grid">
          <label>Dataset ZIP<input name="dataset_zip" type="file" accept=".zip" required></label>
          <label>Model path<input name="model_path" placeholder="checkpoints/channel2_uploaded_cilia_unet.pt"></label>
          <label>Threshold<input name="threshold" type="number" min="0.01" max="0.99" step="0.01" value="0.5"></label>
          <label>Tile size<input name="tile_size" type="number" min="64" step="32" value="512"></label>
          <label>Overlap<input name="overlap" type="number" min="0" step="16" value="64"></label>
          <label>Minimum object size<input name="min_object_size" type="number" min="1" step="1" value="8"></label>
        </div>
        <div class="actions">
          <button class="primary" type="submit">Run Test</button>
        </div>
      </form>
      <pre id="evaluateOutput"></pre>
    </section>

    <section id="train-paired">
      <h2>Train From Image And Mask Folders</h2>
      <div class="note warn">ZIP layout: <code>images/sample_001.tif</code> and <code>masks/sample_001.tif</code>. This uses patch-based training so full-size images are not shrunk before learning. Training will not start unless confirmation is checked.</div>
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
      <pre id="trainPairedOutput"></pre>
    </section>

    <section id="train-roi">
      <h2>Train From ImageJ ROIs</h2>
      <div class="note warn">Upload analyzed TIFFs that contain ImageJ/FIJI overlay ROIs. The app converts ROIs to masks, creates patches, then trains.</div>
      <form id="trainRoiForm">
        <div class="grid">
          <label>Analyzed TIFF ZIP<input name="image_zip" type="file" accept=".zip" required></label>
          <label>Save model path<input name="save_model_path" placeholder="checkpoints/my_roi_cilia_model.pt"></label>
          <label>Epochs<input name="epochs" type="number" min="1" value="10" required></label>
          <label>Batch size<input name="batch_size" type="number" min="1" value="4" required></label>
          <label>Learning rate<input name="learning_rate" type="number" min="0.000001" step="0.000001" value="0.0001"></label>
          <label>Validation fraction<input name="val_fraction" type="number" min="0" max="0.9" step="0.05" value="0.2"></label>
          <label>Patch size<input name="patch_size" type="number" min="64" step="32" value="256"></label>
          <label>Stride<input name="stride" type="number" min="1" step="16" value="128"></label>
          <label>ImageJ label channel<input name="label_channel" type="number" min="1" value="2"></label>
          <label>ROI line width<input name="line_width" type="number" min="1" value="5"></label>
          <label class="check"><input name="confirm_train" type="checkbox"> Confirm training</label>
          <label class="check"><input name="overwrite_model" type="checkbox"> Allow overwrite</label>
        </div>
        <div class="actions">
          <button class="primary" type="submit">Extract ROIs And Train</button>
        </div>
      </form>
      <pre id="trainRoiOutput"></pre>
    </section>

    <section id="extract-rois">
      <h2>Extract ROI Masks</h2>
      <div class="note">Use this to check whether the app can see the ImageJ/FIJI ROI labels inside your analyzed TIFFs before training.</div>
      <form id="extractRoisForm">
        <div class="grid">
          <label>Analyzed TIFF ZIP<input name="image_zip" type="file" accept=".zip" required></label>
          <label>ImageJ label channel<input name="label_channel" type="number" min="1" value="2"></label>
          <label>ROI line width<input name="line_width" type="number" min="1" value="5"></label>
        </div>
        <div class="actions">
          <button class="primary" type="submit">Extract ROI Masks</button>
        </div>
      </form>
      <pre id="extractRoisOutput"></pre>
      <div id="extractRoisPreview" class="preview"></div>
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

    document.getElementById("predictForm").addEventListener("submit", async event => {
      event.preventDefault();
      const payload = await postForm("/api/predict", event.target, "predictOutput");
      document.getElementById("predictPreview").innerHTML = [
        payload.overlay_png_url ? imageFigure(payload.overlay_png_url, "Channel 2 Input + Predicted Cilia Mask") : "",
        payload.mask_png_url ? imageFigure(payload.mask_png_url, "Predicted Cilia Mask") : "",
      ].join("");
    });

    document.getElementById("evaluateForm").addEventListener("submit", async event => {
      event.preventDefault();
      await postForm("/api/evaluate", event.target, "evaluateOutput");
    });

    document.getElementById("trainPairedForm").addEventListener("submit", async event => {
      event.preventDefault();
      await postForm("/api/train-paired", event.target, "trainPairedOutput");
    });

    document.getElementById("trainRoiForm").addEventListener("submit", async event => {
      event.preventDefault();
      await postForm("/api/train-roi", event.target, "trainRoiOutput");
    });

    document.getElementById("extractRoisForm").addEventListener("submit", async event => {
      event.preventDefault();
      const payload = await postForm("/api/extract-rois", event.target, "extractRoisOutput");
      const figures = (payload.images || []).slice(0, 12).map(row => imageFigure(row.overlay_url, row.image));
      document.getElementById("extractRoisPreview").innerHTML = figures.join("");
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
