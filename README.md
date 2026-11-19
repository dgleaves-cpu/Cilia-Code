# Cilia Detector

Local tools for training, evaluating, and using a PyTorch U-Net to segment cilia in multichannel microscopy images. The project is designed around one explicit data rule: the model uses **only the second image channel** (zero-based `CHANNEL_INDEX = 1`) for training, testing, prediction, and display.

The primary interface is a local browser app for researchers. Command-line tools, a Colab notebook, and a lightweight API are included for reproducible or automated workflows.

## What It Does

- Trains or continues training a one-channel binary cilia-segmentation U-Net from paired images and masks.
- Predicts masks for one or more uploaded multichannel images, including multi-image TIFF files.
- Tests a saved model against held-out masks with pixel, object-level, threshold-sweep, ROC/AUC, and precision-recall metrics.
- Analyzes binary masks to measure cilia morphology and create exploratory t-SNE plots.
- Exports organized masks, CSV files, plots, and failure-review figures under `outputs/app/`.

All U-Net inputs have shape `(height, width, 1)`. Other image channels are never merged, averaged, normalized, displayed, or sent to the model.

## Quick Start

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your hardware if the default package installation does not provide one. Apple Silicon machines can use the MPS backend when it is available; the app can fall back to CPU when necessary.

### 2. Start the browser app

```bash
python scripts/cilia_app.py
```

Open <http://127.0.0.1:8000> in a browser. If that port is already in use, start the app on another port:

```bash
python -m uvicorn scripts.cilia_app:app --host 127.0.0.1 --port 8001
```

The browser app is the recommended starting point. It provides these workflows:

| Tab | Purpose |
| --- | --- |
| **Status** | Shows device, checkpoint, folders, and supported formats. |
| **Predict** | Produces masks for one or more multichannel images and exports collective morphology/t-SNE outputs. |
| **Test** | Evaluates a saved U-Net against paired held-out images and masks. |
| **Analyze Masks** | Measures binary masks and creates exploratory morphology/t-SNE outputs without running a model. |
| **Train From Masks** | Trains or continues training from paired multichannel images and manual binary masks. |

## Data Conventions

### Training and test data

Images and masks are paired by **filename stem**. Extensions may differ, but the names before the extension must match. For example, `sample_001.tif` pairs with `sample_001.png`.

```text
paired_cilia_data.zip
  images/
    sample_001.tif
    sample_002.tif
  masks/
    sample_001.tif
    sample_002.tif
```

The app also accepts `training_images/` with `training_masks/`, and `test_images/` with `test_masks/`, when those are the folder names inside a ZIP. It reports missing image/mask pairs before training or testing begins.

Requirements:

- **Images:** supported TIFF/TIF, NPY, NPZ, PNG, JPG, or JPEG files. Multichannel training, testing, and prediction images must contain at least two channels and use a recognizable HWC or CHW layout. TIFF prediction uploads can include several images in a time/scene series.
- **Input channel:** only channel index `1`, the second channel, is selected. It is normalized and processed independently.
- **Masks:** 2D binary masks where cilia are nonzero (`1`) and background is zero. The app converts nonzero mask values to binary cilia labels.
- **Splits:** keep test images separate from images used for model selection and training.

### Model checkpoints

U-Net checkpoints are PyTorch `.pt` files. The app saves metadata that records the one-channel input, image size, threshold, and `channel_index = 1`. A checkpoint trained on a different channel is intentionally rejected to prevent accidental use of the wrong microscopy signal.

Store checkpoints under `checkpoints/`. They are excluded from Git because they can be large and may contain study-specific data.

## Outputs

Each browser-app run receives a timestamped directory below `outputs/app/`. Depending on the workflow, it can contain:

- predicted binary masks and a ZIP/manifest for batch prediction;
- training history CSV plus loss and validation-metric plots;
- test metrics, threshold sweep, ROC/AUC, precision-recall/AP, and failure-review figures;
- mask morphology tables and t-SNE coordinate/plot files.

Mask visualizations use a white background with red cilia. Testing and failure-review figures distinguish true-positive, false-positive, and false-negative objects with circles.

## Repository Map

```text
cilia_detector/        Core image I/O, channel selection, U-Net, morphology, and helpers
scripts/               Browser app, API, command-line tools, and cluster example
notebooks/             Google Colab notebook
tests/                 Automated regression tests
docs/                  Architecture and command-line reference
checkpoints/           Local model checkpoints (created locally; ignored by Git)
outputs/               Local run artifacts (created locally; ignored by Git)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the code map and [docs/COMMAND_LINE_TOOLS.md](docs/COMMAND_LINE_TOOLS.md) for the non-browser workflows.

## Command-Line and API Workflows

The browser app covers the normal local workflow. These optional tools are useful for scripting:

```bash
# Predict one image from a compatible Channel-2 U-Net checkpoint.
python scripts/predict.py \
  --checkpoint checkpoints/my_cilia_model.pt \
  --image data/new_image.tif \
  --out outputs/new_image_mask.tif

# Run the standalone FastAPI service for notebooks or other clients.
python scripts/serve_channel2_api.py
```

The standalone API exposes `GET /health`, `POST /predict`, and `POST /train`. Its `/train` route requires explicit epoch/batch-size values and `confirm_train=true`; it will not silently overwrite a model. See [COLAB_GUIDE.md](COLAB_GUIDE.md) and [docs/COMMAND_LINE_TOOLS.md](docs/COMMAND_LINE_TOOLS.md) for details.

## Google Colab

The complete notebook is [notebooks/channel2_cilia_unet_colab.ipynb](notebooks/channel2_cilia_unet_colab.ipynb). It includes paired data loading, one-channel U-Net training, metrics, prediction visualizations, and the standalone FastAPI service with a temporary public tunnel. The compact setup notes are in [COLAB_GUIDE.md](COLAB_GUIDE.md).

## Development

Run the regression suite from the repository root:

```bash
MPLCONFIGDIR=.build/matplotlib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover the second-channel-only guarantee, TIFF stack handling, batch prediction export, mask-only analysis, and training-device recovery. See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions.

## Scientific Use Note

Retraining requires manually prepared paired cilia masks. Prediction requires only new multichannel images. t-SNE is an exploratory morphology visualization, not a segmentation-accuracy metric; evaluate segmentation quality with held-out masks and the reported pixel/object metrics.
