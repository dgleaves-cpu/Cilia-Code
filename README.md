# Cilia Segmentation

Starter pipeline for denoising the second cilia-marker channel and training a one-channel U-Net to segment cilia.

The reference image stack is an ImageJ TIFF with five planes:

| Plane | Biological channel | Use in this pipeline |
| --- | --- | --- |
| 1 | Axoneme / acetylated tubulin | Not used by the revised U-Net input |
| 2 | Ciliary membrane / ARL13B | Only U-Net input channel, denoised; primary label source |
| 3 | DAPI nuclei | Optional future cell-cilium assignment |
| 4 | mScarlet-KIF3B motor | Downstream intensity classification |
| 5 | mNeonGreen-KIF3A motor | Downstream intensity classification |

The uploaded CSV/Numbers file contains measurements, while the analyzed TIFF contains ImageJ overlay ROIs. `prepare_dataset.py` extracts those ROIs into binary masks and saves training patches.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, install PyTorch using the current command from pytorch.org if the generic wheel is not selected automatically.

## Prepare Training Patches

Put your analyzed TIFFs in one folder. The U-Net pipeline extracts only zero-based channel index `1`, meaning the second channel. If you have matching CSV files, put them in another folder using the same stem, for example:

Regenerate patches after this channel-2-only update. Older `data/patches_*` folders created by the previous two-channel workflow should not be reused for one-channel training.

```text
data/raw/CS02-001-analyzed-80cells.tif
data/results/CS02-001-results.csv
```

Then run:

```bash
python scripts/prepare_dataset.py \
  --image-dir data/raw \
  --out-dir data/patches \
  --channel-index 1 \
  --patch-size 256 \
  --stride 128
```

For the uploaded sample:

```bash
python scripts/prepare_dataset.py \
  --image-dir "/Users/davingleaves/Downloads" \
  --glob "CS02-001-analyzed-80cells.tif" \
  --channel-index 1 \
  --out-dir data/patches_sample
```

## Train U-Net

```bash
python scripts/train_unet.py \
  --patch-dir data/patches \
  --out checkpoints/cilia_unet.pt \
  --epochs 80 \
  --batch-size 8
```

The model is implemented in PyTorch in [cilia_detector/model.py](cilia_detector/model.py). It uses only denoised zero-based channel index `1` as input and learns a one-channel binary cilium mask from the channel-2 ImageJ cilium ROIs.

## Predict On New Images

```bash
python scripts/predict.py \
  --checkpoint checkpoints/cilia_unet.pt \
  --image data/raw/CS02-021.tif \
  --channel-index 1 \
  --out outputs/CS02-021_predicted_cilia_mask.tif
```

For a folder of images:

```bash
python scripts/batch_predict.py \
  --checkpoint checkpoints/cilia_unet.pt \
  --image-dir data/raw \
  --glob "CS02-*-analyzed*.tif" \
  --out-dir outputs/unet_predictions
```

## Export Existing ROI Labels

The analyzed TIFFs already contain ImageJ overlay ROIs. Export those masks and overlays before training so you can inspect the labels:

```bash
python scripts/export_roi_masks.py \
  --image-dir data/raw \
  --glob "CS02-*-analyzed*.tif" \
  --channel-index 1 \
  --out-dir outputs/roi_reference
```

## Colab Notebook And API

The full beginner-friendly notebook is [channel2_cilia_unet_colab.ipynb](notebooks/channel2_cilia_unet_colab.ipynb). It includes:

- `IMAGE_DIR`, `MASK_DIR`, `IMAGE_SIZE`, `CHANNEL_INDEX = 1`, and `MODEL_PATH` configuration variables.
- A paired image/mask loader that validates matching filename stems.
- One-channel PyTorch U-Net training with BCE + Dice loss, Dice coefficient, and IoU.
- Visualizations limited to `Channel 2 Input`, `True Cilia Mask`, and `Predicted Cilia Mask`.
- A FastAPI service with `GET /health`, `POST /predict`, and `POST /train`.

## Local Browser App

Run the app when you want a point-and-click workflow for prediction, testing, training, and ROI label inspection:

```bash
python scripts/cilia_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

The app includes tabs for:

- Status: checks the selected model, project folders, device, and supported formats.
- Predict: uploads one new multichannel image and outputs a predicted cilia mask, channel-2 overlay, and object table when possible.
- Test: uploads a ZIP with paired `images/` and `masks/` folders and reports Dice and IoU.
- Train From Masks: trains a one-channel U-Net from paired multichannel images and binary cilia masks using full-resolution patches by default.
- Train From ImageJ ROIs: extracts embedded ImageJ/FIJI overlay ROIs from analyzed TIFFs, creates mask patches, and trains.
- Extract ROI Masks: exports ROI masks and overlays so you can inspect whether the TIFF labels are being read correctly.

Training ZIPs with prepared masks should look like:

```text
training_pairs.zip
  images/
    CS02-001.tif
    CS02-002.tif
  masks/
    CS02-001.tif
    CS02-002.tif
```

ROI training ZIPs should contain analyzed `.tif` or `.tiff` files with ImageJ/FIJI overlay ROIs embedded in them. All app model input uses only zero-based channel index `1`, meaning the second image channel.

For large images with tiny cilia labels, use the app's patch-based defaults instead of shrinking the whole image:

```text
Patch size: 512
Stride: 256
Minimum positive pixels: 3
Background patches per positive patch: 1.0
```

This preserves small cilia masks from `2048 x 2048` images and gives the U-Net many local examples. A full `2048 x 2048` image resized to `256 x 256` can reduce a labeled cilium to only a few pixels, which usually leads to near-zero Dice.

## Classify And Measure Cilia

After prediction, classify connected cilia objects and measure support from channels 1, 2, 4, and 5:

```bash
python scripts/classify_cilia.py \
  --image data/raw/CS02-021.tif \
  --mask outputs/CS02-021_predicted_cilia_mask.tif \
  --out outputs/CS02-021_cilia_objects.csv
```

The first-pass classifier is intentionally transparent: it flags likely cilia using object area, skeleton length, ARL13B intensity, axoneme support, and optional KIF3A/KIF3B intensity. Once you have enough verified predictions, replace the rule thresholds with a small object-level classifier.

## ISAAC / SLURM

Edit `scripts/isaac_train.slurm` for your module names and project path, then submit:

```bash
sbatch scripts/isaac_train.slurm
```

## Google Colab

Upload this folder to Drive, open [channel2_cilia_unet_colab.ipynb](notebooks/channel2_cilia_unet_colab.ipynb), and run the cells top to bottom. Colab already includes PyTorch on GPU runtimes, so installation may only need the image-processing and API packages. See [COLAB_GUIDE.md](COLAB_GUIDE.md) for a compact checklist.
