# Cilia U-Net Optimization Protocol

This document describes optional, reproducible experiments run with `scripts/optimize_cilia.py`. The experiment runner writes a new timestamped directory below `outputs/optimization_20260910/`; it does not overwrite existing checkpoints.

## Goal

The optimization workflow uses an image-level train/validation split before patches are created. This prevents patches from one source image appearing in both partitions, which can make patch-level validation look stronger than generalization to a new image.

Select an experiment and probability threshold with validation images. Keep a separate, untouched test set for the final report.

## Suggested Stages

1. **Baseline:** BCE + Dice loss, flips, and the current patch settings.
2. **Hard-negative sampling:** retain the loss while increasing background patches per positive patch to target bright non-cilia structures.
3. **False-positive-focused loss:** compare the better candidate with BCE + Tversky, using a larger false-positive penalty. Add rotations or mild intensity jitter only when the baseline comparison identifies a need.

Change one or two parameters at a time and record the complete configuration saved with every run. Do not select a threshold from an image that will later be claimed as a final independent test image.

## Example Baseline

Replace the placeholder folders with paired source image and mask directories. File names must share their stem.

```bash
MPLCONFIGDIR=/private/tmp/cilia_matplotlib .venv/bin/python scripts/optimize_cilia.py run \
  --images /path/to/training_images \
  --masks /path/to/training_masks \
  --name baseline_source_split \
  --epochs 20 --batch-size 4 --learning-rate 0.0001 \
  --patch-size 512 --stride 256 --min-positive-pixels 15 \
  --background-ratio 1.5 --loss bce_dice
```

## Example False-Positive Candidate

```bash
MPLCONFIGDIR=/private/tmp/cilia_matplotlib .venv/bin/python scripts/optimize_cilia.py run \
  --images /path/to/training_images \
  --masks /path/to/training_masks \
  --name fp_tversky_source_split \
  --epochs 20 --batch-size 4 --learning-rate 0.0001 \
  --patch-size 512 --stride 256 --min-positive-pixels 15 \
  --background-ratio 2.5 --loss bce_tversky \
  --tversky-alpha 0.70 --tversky-beta 0.30 --rot90 --intensity
```

To compare completed runs:

```bash
.venv/bin/python scripts/optimize_cilia.py compare \
  outputs/optimization_20260910/<baseline-id> \
  outputs/optimization_20260910/<candidate-id>
```

Each run records its source split, patch counts, configuration, history, checkpoint, threshold sweep, pixel metrics, object table, false-positive analysis, and per-image reviews. Review the lowest-performing images before changing the next experiment.
