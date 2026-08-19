#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cilia_detector.channel2_pipeline import (
    CHANNEL_INDEX,
    IMAGE_SIZE,
    MODEL_PATH,
    SUPPORTED_IMAGE_EXTENSIONS,
    find_image_mask_pairs,
    load_paired_training_images_and_masks,
    load_unet_model,
    png_bytes_from_mask,
    png_bytes_from_overlay,
    predict_preprocessed,
    preprocess_image_file,
    train_unet_from_arrays,
)


app = FastAPI(
    title="Channel 2 Cilia Segmentation API",
    description=(
        "Binary cilia segmentation for multichannel bio-images. The API always "
        "extracts zero-based channel index 1 and never uses other channels for prediction."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "channel_index": CHANNEL_INDEX,
        "image_size": IMAGE_SIZE,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_overlay: bool = Form(True),
):
    """Predict a binary cilia mask from one multichannel image.

    Supported inputs: TIFF/TIF, NPY, NPZ, PNG, JPG/JPEG. The image must be HWC
    or CHW and must contain at least two channels because CHANNEL_INDEX=1 is
    required.
    """
    if threshold <= 0 or threshold >= 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1, for example 0.5.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{suffix}'. Supported formats: "
                f"{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
            ),
        )

    try:
        model, checkpoint = load_unet_model(MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load the U-Net model: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="cilia_predict_") as tmp:
        image_path = Path(tmp) / f"uploaded{suffix}"
        with image_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        try:
            image_size = tuple(checkpoint.get("image_size", IMAGE_SIZE))
            channel = preprocess_image_file(image_path, image_size=image_size, channel_index=CHANNEL_INDEX)
            _, mask = predict_preprocessed(model, channel, threshold=threshold)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Prediction failed. Check that the file is a multichannel bio-image in HWC or CHW order "
                    f"with at least two channels. Details: {exc}"
                ),
            ) from exc

        mask_png = png_bytes_from_mask(mask)
        if not return_overlay:
            return StreamingResponse(BytesIO(mask_png), media_type="image/png")

        overlay_png = png_bytes_from_overlay(channel, mask)
        archive = BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("predicted_cilia_mask.png", mask_png)
            zf.writestr("channel2_predicted_overlay.png", overlay_png)
        archive.seek(0)
        return StreamingResponse(
            archive,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="cilia_prediction.zip"'},
        )


@app.post("/train")
async def train(
    zip_file: UploadFile = File(...),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    learning_rate: float = Form(1e-4),
    val_fraction: float = Form(0.2),
    confirm_train: bool = Form(False),
    save_model_path: str | None = Form(None),
    overwrite_model: bool = Form(False),
):
    """Fine-tune the existing one-channel U-Net from a ZIP of paired images/masks.

    ZIP layout:
      images/<same_stem>.tif
      masks/<same_stem>.tif
    """
    if not confirm_train:
        raise HTTPException(
            status_code=400,
            detail="Training was not started. Resubmit with confirm_train=true, epochs, and batch_size.",
        )
    if epochs <= 0:
        raise HTTPException(status_code=400, detail="epochs must be a positive integer.")
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be a positive integer.")
    if learning_rate <= 0:
        raise HTTPException(status_code=400, detail="learning_rate must be positive.")
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Existing model not found at {MODEL_PATH}. Train an initial model before using /train fine-tuning.",
        )

    target_model = Path(save_model_path) if save_model_path else Path(
        f"checkpoints/channel2_cilia_unet_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    )
    if target_model.exists() and not overwrite_model:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Refusing to overwrite existing model {target_model}. "
                "Set overwrite_model=true or provide a new save_model_path."
            ),
        )

    with tempfile.TemporaryDirectory(prefix="cilia_train_") as tmp:
        tmp_path = Path(tmp)
        uploaded_zip = tmp_path / "training_data.zip"
        with uploaded_zip.open("wb") as handle:
            shutil.copyfileobj(zip_file.file, handle)
        try:
            extract_dir = tmp_path / "training_data"
            extract_dir.mkdir()
            safe_extract_zip(uploaded_zip, extract_dir)
            image_dir, mask_dir = find_training_dirs(extract_dir)
            pairs = find_image_mask_pairs(image_dir, mask_dir)
            images, masks, _ = load_paired_training_images_and_masks(
                image_dir=image_dir,
                mask_dir=mask_dir,
                image_size=IMAGE_SIZE,
                channel_index=CHANNEL_INDEX,
            )
            model, _ = load_unet_model(MODEL_PATH)
            _, history = train_unet_from_arrays(
                images,
                masks,
                model=model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_fraction=val_fraction,
                model_path=target_model,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Training failed. Expected a ZIP with images/ and masks/ folders, matching filename stems, "
                    f"multichannel images, and binary masks. Details: {exc}"
                ),
            ) from exc

    return JSONResponse(
        {
            "status": "ok",
            "message": "Fine-tuning completed using only zero-based channel index 1.",
            "trained_pairs": len(pairs),
            "saved_model_path": str(target_model),
            "final_metrics": history[-1] if history else {},
            "history": history,
        }
    )


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise HTTPException(status_code=400, detail="ZIP file contains an unsafe path.")
        zf.extractall(destination)


def find_training_dirs(extract_dir: Path) -> tuple[Path, Path]:
    candidates = [extract_dir]
    candidates.extend([p for p in extract_dir.iterdir() if p.is_dir()])
    for base in candidates:
        image_dir = base / "images"
        mask_dir = base / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            return image_dir, mask_dir
    raise HTTPException(
        status_code=400,
        detail="Could not find images/ and masks/ folders inside the ZIP file.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
