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

### Optional pretrained comparison models

The browser app can compare a saved Channel-2 U-Net against pretrained Cellpose-SAM v2 and Omnipose models. Install those optional backends once in the same environment as the app:

```bash
bash scripts/install_model_comparison_backends.sh
```

The first comparison that selects a pretrained model may download model weights, so it requires an internet connection once. Afterwards, cached weights can be used offline. These pretrained models are exploratory baselines, not cilia-specific replacements for your labeled-data U-Net.

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
  --patch-size 256 \
  --stride 128
```

For the uploaded sample:

```bash
python scripts/prepare_dataset.py \
  --image-dir "/Users/davingleaves/Downloads" \
  --glob "CS02-001-analyzed-80cells.tif" \
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
- Test: uploads a ZIP with paired `images/` and `masks/` folders (or `test_images/` and `test_masks/`) and reports Dice and IoU.
- Analyze Masks: uploads a ZIP of binary manual cilia masks and creates mask-only morphology measurements plus exploratory t-SNE plots for cilia length and curvature. It does not run a model or use microscopy-image channels.
- Compare Labkit + U-Net: runs the selected Channel-2-only U-Net on a paired test ZIP and compares its binary masks with an aligned Labkit TIFF export, manual masks, and object-level measurements.
- Test Cellpose/Omnipose: tests exactly one selected pretrained or custom Cellpose/Omnipose model against held-out masks, with its own shared pixel and object metrics plus failure-review image.
- Compare Models: runs one unchanged paired test ZIP through one or two saved Channel-2 U-Net checkpoints, pretrained Cellpose-SAM v2 and Omnipose baselines, plus optional custom Cellpose and Omnipose checkpoints. It shows shared pixel Dice/IoU and object precision/recall/F1 side by side, with a per-model failure-review image. It never trains or changes any checkpoint.
- Improve Model: runs a source-image-disjoint optimization experiment. It offers a clean baseline followed by a focused false-positive profile with more background patches and BCE + Tversky loss. It selects thresholds on validation images only and preserves all existing checkpoints.
- Train From Masks: trains a one-channel U-Net from paired multichannel images and binary cilia masks using full-resolution patches by default.
- Train Omnipose: fine-tunes a separate one-channel Omnipose instance-segmentation model from the same paired cilia masks. It uses only Channel 2, runs on CPU for stable Mac training, saves a separate model file and loss history, and can be added to Compare Models by its saved path.
- Train Cellpose: fine-tunes the pretrained Cellpose-SAM v2 instance-segmentation model from the same paired cilia masks. It reads only Channel 2, runs on CPU for stable Mac training, saves a separate model file and loss history, and can be added to Compare Models by its saved path. The installed Cellpose-SAM release requires a pretrained starting model. It internally pads the single selected input plane with zero-filled channels required by its architecture; this does not read, merge, or display any other microscopy channel.
- Train From ImageJ ROIs: extracts embedded ImageJ/FIJI overlay ROIs from analyzed TIFFs, creates mask patches, and trains.
- Extract ROI Masks: exports ROI masks and overlays so you can inspect whether the TIFF labels are being read correctly.

Training runs in the background, so the page stays responsive and shows the current patch-preparation stage, each completed epoch's loss/Dice/IoU, percentage, elapsed time, and the finished plots.

The custom Omnipose workflow is separate from U-Net training. It starts from a single-channel Omnipose base model by default, or can train from scratch. Its model learns connected cilium instances rather than a U-Net probability map, so add its saved path in **Compare Models** to evaluate it using the same pixel Dice/IoU and object metrics as the other methods.

The custom Cellpose workflow is similarly separate from U-Net training. It fine-tunes Cellpose-SAM from connected cilia objects in your masks, records the training and validation loss history, and saves a checkpoint path that can be pasted into **Compare Models**. Cellpose and Omnipose are instance-segmentation methods, so the comparison reports shared binary-mask and object metrics rather than U-Net probability-map ROC/AUC or average precision.

For a reliable experiment, use **Improve Model** before the regular Test screen. The optimization flow holds out three complete source images before it creates patches, keeps validation unaugmented, records the experiment settings and selected threshold, and produces threshold, ROC, precision-recall, and false-positive analysis figures. The Test screen should be reserved for final/exploratory evaluation rather than choosing a threshold.

Training ZIPs with prepared masks can use either `images/` and `masks/`, or `training_images/` and `training_masks/`:

```text
training_pairs.zip
  images/
    CS02-001.tif
    CS02-002.tif
  masks/
    CS02-001.tif
    CS02-002.tif
```

For **Analyze Masks**, use the mask folder by itself. A ZIP containing only supported mask image files also works:

```text
cilia_masks.zip
  masks/
    CS02-001.tif
    CS02-002.tif
```

For **Compare Labkit + U-Net**, upload the same paired TIFF test ZIP used on the Test screen and a Labkit binary TIFF export. A single TIFF is used for a one-image test set; for multiple images, export a TIFF stack with one binary page per image. The pages must follow the same alphabetical order as the image filenames in the test ZIP. The results include each method's Dice/IoU/object metrics against manual masks, their direct agreement, a per-image CSV, and aligned visual reviews.

ROI training ZIPs should contain analyzed `.tif` or `.tiff` files with cilia ROIs assigned to ImageJ channel `2`. All app model input uses only zero-based channel index `1`, meaning the second image channel. The app does not fall back to ROI labels from another channel, and it rejects a checkpoint that does not record `channel_index = 1`.

For large images with tiny cilia labels, use the app's patch-based defaults instead of shrinking the whole image:

```text
Patch size: 512
Stride: 256
Minimum positive pixels: 3
Background patches per positive patch: 1.0
```

This preserves small cilia masks from `2048 x 2048` images and gives the U-Net many local examples. A full `2048 x 2048` image resized to `256 x 256` can reduce a labeled cilium to only a few pixels, which usually leads to near-zero Dice.

## Analysis Outputs

The browser app saves every plot and its supporting CSV in the run's folder under `outputs/app/`:

- Training saves `training_loss.png`, `validation_metrics.png`, and `training_history.csv`.
- Testing saves pixel ROC/AUC and precision-recall/AP graphs from the manual test masks, plus the per-image Dice/IoU table. ROC uses a balanced sample for ranking; precision-recall uses weighted sampling to preserve the true cilia prevalence.
- Testing also saves a threshold sweep for Dice/IoU, object-level precision/recall/F1 based on connected-object IoU matching, and a failure-review gallery ordered from the lowest Dice image upward. Each gallery image contains only Channel 2 Input, True Cilia Mask, and Predicted Cilia Mask.
- Prediction and testing save cilium morphology measurements. With at least four measured objects, the app creates t-SNE plots colored by length and curvature proxy. The curvature proxy is tortuosity: skeleton path length divided by endpoint distance, where `1` is approximately straight and larger values are more curved.
- Analyze Masks produces the same morphology CSV and t-SNE plots directly from your manual masks, along with a per-mask summary CSV. It is useful for checking the range and outliers in the label set before training.

t-SNE is exploratory rather than a model-accuracy score. Use it to find groups or outliers for manual review, not as evidence that the model is accurate.

For a fair U-Net/Cellpose/Omnipose comparison, upload a held-out TIFF test ZIP with the same structure as the training example above. The comparison workflow extracts, denoises, and passes only zero-based image channel `1` to every model. It reports only metrics all methods share: pixel Dice, pixel IoU, and connected-object precision, recall, and F1. Cellpose and Omnipose return instance masks rather than calibrated per-pixel probability maps, so ROC/AUC and average precision are not shown for those two methods.

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
