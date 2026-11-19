# Command-Line Tools

The local browser app (`python scripts/cilia_app.py`) is the recommended workflow. The scripts below are optional tools for repeatable command-line, cluster, or programmatic work.

All U-Net scripts require a compatible one-channel checkpoint trained with `channel_index = 1`. They use only the second image channel.

## Prediction

Predict a binary mask for one image:

```bash
python scripts/predict.py \
  --checkpoint checkpoints/my_cilia_model.pt \
  --image data/new_image.tif \
  --out outputs/new_image_mask.tif \
  --threshold 0.90
```

`--tile-size` and `--overlap` control tiled full-resolution prediction. The optional `--channel-index` argument remains available for compatibility, but it must be `1`.

Predict a folder and create per-image object tables:

```bash
python scripts/batch_predict.py \
  --checkpoint checkpoints/my_cilia_model.pt \
  --image-dir data/images \
  --glob "*.tif" \
  --out-dir outputs/batch_prediction
```

`batch_predict.py` calls `predict.py` and then `classify_cilia.py` for each image.

## Patch-Based Legacy Training

`scripts/train_unet.py` trains from an existing folder of `.npz` image/mask patches. This is useful when a patch dataset has already been prepared for a cluster workflow. The browser app is preferred for creating and training from paired full-resolution images and masks.

```bash
python scripts/train_unet.py \
  --patch-dir data/patches \
  --out checkpoints/my_cilia_model.pt \
  --epochs 80 \
  --batch-size 8 \
  --lr 0.0001
```

Each patch archive must contain an `image` array shaped `(1, height, width)` and a 2D `mask` array. The script adds Channel-2 checkpoint metadata so its output can be loaded by the browser app.

## Standalone API

Run the small FastAPI service used by the Colab notebook:

```bash
python scripts/serve_channel2_api.py
```

It listens on port `8000` and provides:

- `GET /health`
- `POST /predict`
- `POST /train`

`POST /train` accepts a ZIP with `images/` and `masks/` directories. It requires explicit `epochs`, `batch_size`, and `confirm_train=true`; existing checkpoints are protected unless `overwrite_model=true` is also sent.

## Optimization Experiments

`scripts/optimize_cilia.py` is an advanced, standalone experiment runner. It creates an image-level source split before patch extraction so that patches from a source image do not cross from training into validation.

```bash
python scripts/optimize_cilia.py run \
  --images /path/to/training_images \
  --masks /path/to/training_masks \
  --name baseline_source_split \
  --epochs 20 --batch-size 4 --learning-rate 0.0001 \
  --patch-size 512 --stride 256 --min-positive-pixels 15 \
  --background-ratio 1.5 --loss bce_dice
```

Use the held-out test arguments only after selecting an experiment and threshold on validation images. A fuller staged protocol is in [../OPTIMIZATION.md](../OPTIMIZATION.md).

## Other Utilities

| Script | Purpose |
| --- | --- |
| `scripts/classify_cilia.py` | Measures predicted connected objects and applies the optional downstream intensity-based classifier. It can use channels beyond channel 2 because it is not an input to the U-Net. |
| `scripts/denoise_image.py` | Writes a denoised selected-channel image for inspection. |
| `scripts/isaac_train.slurm` | Editable SLURM template for training from a pre-existing patch directory. |

Run `python <script> --help` for the complete arguments of a given command-line tool.
