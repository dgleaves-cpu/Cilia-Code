from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from skimage.transform import resize
import tifffile as tiff
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .model import UNet, dice_loss


IMAGE_DIR = Path("data/images")
MASK_DIR = Path("data/masks")
IMAGE_SIZE = (256, 256)
CHANNEL_INDEX = 1
MODEL_PATH = Path("checkpoints/channel2_cilia_unet.pt")

SUPPORTED_IMAGE_EXTENSIONS = {".tif", ".tiff", ".npy", ".npz", ".png", ".jpg", ".jpeg"}
SUPPORTED_MASK_EXTENSIONS = {".tif", ".tiff", ".npy", ".npz", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class PairRecord:
    stem: str
    image_path: Path
    mask_path: Path


def coerce_image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("IMAGE_SIZE must be an int or a tuple like (height, width).")
    return (int(image_size[0]), int(image_size[1]))


def read_array(path: str | Path) -> np.ndarray:
    """Read TIFF, NumPy, or common image files into a NumPy array."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return np.asarray(tiff.imread(path))
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"{path.name} is an .npz with multiple arrays. Use a single-array .npz.")
        return np.asarray(data[data.files[0]])
    if suffix in {".png", ".jpg", ".jpeg"}:
        return np.asarray(Image.open(path))
    raise ValueError(
        f"Unsupported file format for {path.name}. Supported formats: "
        f"{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
    )


def standardize_multichannel_to_hwc(image: np.ndarray) -> np.ndarray:
    """Return image as (height, width, channels), without combining channels.

    Supported channel orders:
    - HWC: height, width, channels
    - CHW: channels, height, width, common for ImageJ multipage TIFFs
    """
    arr = np.asarray(image)
    if arr.ndim > 3:
        squeezed = np.squeeze(arr)
        if squeezed.ndim != arr.ndim:
            arr = squeezed

    if arr.ndim == 2:
        return arr[..., np.newaxis]
    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 2D image or a 3D multichannel image, got shape {arr.shape}. "
            "Supported orders are HWC and CHW. If your TIFF is Z/C/Y/X or T/C/Y/X, "
            "export a single plane first."
        )

    if arr.shape[-1] <= 16:
        return arr
    if arr.shape[0] <= 16:
        return np.moveaxis(arr, 0, -1)
    raise ValueError(
        f"Could not infer the channel axis for image shape {arr.shape}. "
        "Expected HWC with a small final channel dimension or CHW with a small first dimension."
    )


def select_channel_preserve_dim(image: np.ndarray, channel_index: int = CHANNEL_INDEX) -> np.ndarray:
    """Select zero-based channel_index and preserve HWC channel dimension.

    This is the core rule for the revised notebook. After standardizing to HWC,
    channel 2 is selected exactly as image[..., 1:2]. No other channel is
    normalized, resized, averaged, merged, displayed, or sent to the model.
    """
    hwc = standardize_multichannel_to_hwc(image)
    if channel_index < 0 or channel_index >= hwc.shape[-1]:
        raise ValueError(
            f"Requested zero-based CHANNEL_INDEX={channel_index}, but image has "
            f"{hwc.shape[-1]} channel(s). Channel 2 input requires at least 2 channels."
        )
    return hwc[..., channel_index : channel_index + 1].astype(np.float32)


def normalize_selected_channel(channel_hwc: np.ndarray, lower: float = 1.0, upper: float = 99.8) -> np.ndarray:
    channel_hwc = channel_hwc.astype(np.float32)
    lo, hi = np.percentile(channel_hwc, (lower, upper))
    if hi <= lo:
        return np.zeros_like(channel_hwc, dtype=np.float32)
    normalized = (channel_hwc - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def preprocess_image_array(
    image: np.ndarray,
    image_size: int | tuple[int, int] = IMAGE_SIZE,
    channel_index: int = CHANNEL_INDEX,
) -> np.ndarray:
    """Select, normalize, and resize only channel index 1.

    Returns an HWC array with shape (height, width, 1). PyTorch later converts
    it to NCHW, but keeping HWC here makes the selected-channel rule explicit.
    """
    height, width = coerce_image_size(image_size)
    channel = select_channel_preserve_dim(image, channel_index=channel_index)
    channel = normalize_selected_channel(channel)
    resized = resize(
        channel,
        (height, width, 1),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    return resized.astype(np.float32)


def preprocess_image_file(
    image_path: str | Path,
    image_size: int | tuple[int, int] = IMAGE_SIZE,
    channel_index: int = CHANNEL_INDEX,
) -> np.ndarray:
    return preprocess_image_array(read_array(image_path), image_size=image_size, channel_index=channel_index)


def preprocess_mask_array(mask: np.ndarray, image_size: int | tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load a binary cilia mask as HWC with cilia=1 and background=0."""
    height, width = coerce_image_size(image_size)
    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[-1] <= 4:
            arr = arr[..., 0]
        elif arr.shape[0] <= 4:
            arr = arr[0]
        else:
            raise ValueError(f"Mask shape {arr.shape} is ambiguous. Use a 2D binary mask.")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got shape {arr.shape}.")
    binary = (arr > 0).astype(np.float32)
    resized = resize(
        binary,
        (height, width),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return (resized > 0.5).astype(np.float32)[..., np.newaxis]


def preprocess_mask_file(mask_path: str | Path, image_size: int | tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    return preprocess_mask_array(read_array(mask_path), image_size=image_size)


def collect_files_by_stem(folder: str | Path, extensions: Iterable[str]) -> dict[str, Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a folder, got: {folder}")
    extensions = {ext.lower() for ext in extensions}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    by_stem: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in files:
        if path.stem in by_stem:
            duplicates.setdefault(path.stem, [by_stem[path.stem].name]).append(path.name)
        by_stem[path.stem] = path
    if duplicates:
        details = "\n".join(f"- {stem}: {', '.join(names)}" for stem, names in duplicates.items())
        raise ValueError(f"Duplicate filename stems found. Keep one file per stem:\n{details}")
    return by_stem


def find_image_mask_pairs(image_dir: str | Path, mask_dir: str | Path) -> list[PairRecord]:
    """Match training images and masks by filename stem."""
    image_map = collect_files_by_stem(image_dir, SUPPORTED_IMAGE_EXTENSIONS)
    mask_map = collect_files_by_stem(mask_dir, SUPPORTED_MASK_EXTENSIONS)

    if not image_map:
        raise FileNotFoundError(f"No supported image files found in {image_dir}.")
    if not mask_map:
        raise FileNotFoundError(f"No supported mask files found in {mask_dir}.")

    missing_masks = [path.name for stem, path in sorted(image_map.items()) if stem not in mask_map]
    if missing_masks:
        missing_text = "\n".join(f"- {name}" for name in missing_masks)
        raise FileNotFoundError(
            "Every training image needs a matching mask with the same filename stem. "
            "For example, images/CS02-001.tif should match masks/CS02-001.tif.\n"
            f"Images missing masks:\n{missing_text}"
        )

    return [
        PairRecord(stem=stem, image_path=image_path, mask_path=mask_map[stem])
        for stem, image_path in sorted(image_map.items())
    ]


def load_paired_training_images_and_masks(
    image_dir: str | Path = IMAGE_DIR,
    mask_dir: str | Path = MASK_DIR,
    image_size: int | tuple[int, int] = IMAGE_SIZE,
    channel_index: int = CHANNEL_INDEX,
) -> tuple[np.ndarray, np.ndarray, list[PairRecord]]:
    """Load paired images/masks from folders by matching filename stems.

    Returns:
    - X: float32 images with shape (N, height, width, 1)
    - y: float32 binary masks with shape (N, height, width, 1)
    - pairs: filenames used for each row
    """
    pairs = find_image_mask_pairs(image_dir, mask_dir)
    images = []
    masks = []
    for pair in pairs:
        images.append(preprocess_image_file(pair.image_path, image_size=image_size, channel_index=channel_index))
        masks.append(preprocess_mask_file(pair.mask_path, image_size=image_size))
    return np.stack(images).astype(np.float32), np.stack(masks).astype(np.float32), pairs


class HWCArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray | None = None, augment: bool = False):
        self.images = images
        self.masks = masks
        self.augment = augment
        if masks is not None and len(images) != len(masks):
            raise ValueError("images and masks must contain the same number of items.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        mask = None if self.masks is None else self.masks[idx]

        if self.augment and mask is not None:
            if np.random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.random() < 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

        image_tensor = torch.from_numpy(np.moveaxis(image, -1, 0)).float()
        if mask is None:
            return image_tensor
        mask_tensor = torch.from_numpy(np.moveaxis(mask, -1, 0)).float()
        return image_tensor, mask_tensor


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    intersection = torch.sum(preds * targets)
    pred_sum = torch.sum(preds)
    target_sum = torch.sum(targets)
    union = pred_sum + target_sum - intersection
    dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return {"dice": float(dice.item()), "iou": float(iou.item())}


def train_unet_from_arrays(
    images: np.ndarray,
    masks: np.ndarray,
    model: UNet | None = None,
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    model_path: str | Path | None = MODEL_PATH,
    device: torch.device | None = None,
) -> tuple[UNet, list[dict[str, float]]]:
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if images.ndim != 4 or images.shape[-1] != 1:
        raise ValueError(f"Expected images with shape (N, H, W, 1), got {images.shape}.")
    if masks.ndim != 4 or masks.shape[-1] != 1:
        raise ValueError(f"Expected masks with shape (N, H, W, 1), got {masks.shape}.")

    device = device or get_device()
    model = model or UNet(in_channels=1)
    model = model.to(device)

    dataset = HWCArrayDataset(images, masks, augment=True)
    if len(dataset) > 1:
        val_size = max(1, int(len(dataset) * val_fraction))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    else:
        train_ds = dataset
        val_ds = dataset

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0], device=device))

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_images, batch_masks in train_loader:
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = bce(logits, batch_masks) + dice_loss(logits, batch_masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * batch_images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        metric_totals = {"dice": 0.0, "iou": 0.0}
        seen = 0
        with torch.no_grad():
            for batch_images, batch_masks in val_loader:
                batch_images = batch_images.to(device)
                batch_masks = batch_masks.to(device)
                logits = model(batch_images)
                loss = bce(logits, batch_masks) + dice_loss(logits, batch_masks)
                batch_n = batch_images.size(0)
                val_loss += float(loss.item()) * batch_n
                metrics = binary_metrics_from_logits(logits, batch_masks)
                metric_totals["dice"] += metrics["dice"] * batch_n
                metric_totals["iou"] += metrics["iou"] * batch_n
                seen += batch_n
        val_loss /= len(val_loader.dataset)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_dice": float(metric_totals["dice"] / max(seen, 1)),
            "val_iou": float(metric_totals["iou"] / max(seen, 1)),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f} "
            f"val_iou={row['val_iou']:.4f}"
        )

    if model_path is not None:
        save_unet_model(model, model_path, image_size=images.shape[1:3], channel_index=CHANNEL_INDEX)
    return model, history


def save_unet_model(
    model: UNet,
    model_path: str | Path = MODEL_PATH,
    image_size: tuple[int, int] = IMAGE_SIZE,
    channel_index: int = CHANNEL_INDEX,
) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": 1,
            "image_size": tuple(int(x) for x in image_size),
            "channel_index": int(channel_index),
            "threshold": 0.5,
        },
        model_path,
    )


def load_unet_model(model_path: str | Path = MODEL_PATH, device: torch.device | None = None) -> tuple[UNet, dict]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    device = device or get_device()
    checkpoint = torch.load(model_path, map_location=device)
    in_channels = int(checkpoint.get("in_channels", 1))
    if in_channels != 1:
        raise ValueError(
            f"This channel-2 notebook requires a one-channel U-Net, but {model_path} "
            f"was saved with in_channels={in_channels}. Retrain with CHANNEL_INDEX=1."
        )
    checkpoint_channel_index = checkpoint.get("channel_index")
    if checkpoint_channel_index != CHANNEL_INDEX:
        raise ValueError(
            f"This app only accepts checkpoints trained on zero-based channel index {CHANNEL_INDEX} "
            f"(the second image channel), but {model_path} records channel_index={checkpoint_channel_index!r}. "
            "Retrain the model with the second-channel-only workflow."
        )
    model = UNet(in_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_preprocessed(
    model: UNet,
    image_hwc: np.ndarray,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if image_hwc.ndim != 3 or image_hwc.shape[-1] != 1:
        raise ValueError(f"Expected a preprocessed image with shape (H, W, 1), got {image_hwc.shape}.")
    device = device or get_device()
    model = model.to(device)
    model.eval()
    tensor = torch.from_numpy(np.moveaxis(image_hwc, -1, 0)[np.newaxis]).float().to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask = (probability >= threshold).astype(np.uint8)
    return probability.astype(np.float32), mask


def png_bytes_from_mask(mask: np.ndarray) -> bytes:
    """Encode a binary cilia mask as red cilia on a white background."""
    import io

    mask_bool = mask.astype(bool)
    rgb = np.full((*mask_bool.shape, 3), 255, dtype=np.uint8)
    rgb[mask_bool] = (214, 39, 40)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def png_bytes_from_overlay(channel_hwc: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> bytes:
    import io

    channel = np.squeeze(channel_hwc)
    if channel.ndim != 2:
        raise ValueError("Overlay expects the already-selected single channel.")
    base = (np.clip(channel, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    mask_bool = mask.astype(bool)
    rgb[mask_bool] = (1 - alpha) * rgb[mask_bool] + alpha * red[mask_bool]
    buffer = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()
