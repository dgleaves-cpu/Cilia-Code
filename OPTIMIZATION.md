# Cilia U-Net Optimization Protocol

This protocol leaves the original application and checkpoints unchanged. New runs
are written below `outputs/optimization_20260910/` with a unique experiment ID.

## Current Assessment

`checkpoints/cilia_trial3.pt` is the legacy baseline. Its existing patch-level
validation is not reliable for model selection because patches from one original
2048 x 2048 image can be placed into both train and validation sets, and the
shared dataset also applies random flips during validation. The 3-image test set
was used to choose 0.90 previously, so it is treated as exploratory until new,
untouched labeled images are available.

The `0.90` diagnostic on that exploratory set found 22 false-positive objects
and 17 matched detections. False positives were generally much smaller and
brighter than matched cilia, but their probabilities were also high. That points
to fluorescent speckles and short textured fragments being learned as cilia-like
structures. It does not justify choosing a higher threshold from this test set.

## Staged Experiments

1. `baseline_source_split`: reproduce the existing U-Net, BCE + Dice loss,
   horizontal/vertical flips, and the current patch settings, but split source
   images before patch generation. This is the clean baseline.
2. `hard_negative_source_split`: retain the baseline loss but make the requested
   background ratio real by sampling additional random, mask-negative patches.
   This directly targets bright background texture.
3. `fp_tversky_source_split`: promote the better of the first two experiments
   and test BCE + Tversky with alpha 0.70 and beta 0.30. The larger alpha gives
   false-positive pixels more penalty. Use 90-degree rotations and mild intensity
   jitter only after the hard-negative result is known.

Select the probability threshold from each experiment's image-level validation
images. Run the external three-image set only after selecting the experiment and
threshold; its result remains exploratory because it was viewed previously.

## Reproducible Commands

Run the clean baseline:

```bash
MPLCONFIGDIR=/private/tmp/cilia_matplotlib .venv/bin/python scripts/optimize_cilia.py run \
  --images "/Users/davingleaves/Desktop/Cilia Data/training_images" \
  --masks "/Users/davingleaves/Desktop/Cilia Data/training_masks" \
  --test-images "/Users/davingleaves/Desktop/Cilia Data/test_images" \
  --test-masks "/Users/davingleaves/Desktop/Cilia Data/test_masks" \
  --name baseline_source_split --epochs 20 --batch-size 4 --learning-rate 0.0001 \
  --patch-size 512 --stride 256 --min-positive-pixels 15 --background-ratio 1.5 \
  --loss bce_dice
```

Run the focused false-positive candidate after the baseline:

```bash
MPLCONFIGDIR=/private/tmp/cilia_matplotlib .venv/bin/python scripts/optimize_cilia.py run \
  --images "/Users/davingleaves/Desktop/Cilia Data/training_images" \
  --masks "/Users/davingleaves/Desktop/Cilia Data/training_masks" \
  --test-images "/Users/davingleaves/Desktop/Cilia Data/test_images" \
  --test-masks "/Users/davingleaves/Desktop/Cilia Data/test_masks" \
  --name fp_tversky_source_split --epochs 20 --batch-size 4 --learning-rate 0.0001 \
  --patch-size 512 --stride 256 --min-positive-pixels 15 --background-ratio 2.5 \
  --loss bce_tversky --tversky-alpha 0.70 --tversky-beta 0.30 --rot90 --intensity
```

Compare completed runs:

```bash
.venv/bin/python scripts/optimize_cilia.py compare \
  outputs/optimization_20260910/<baseline-id> \
  outputs/optimization_20260910/<candidate-id>
```

Each run writes its source split, patch sampling counts, full configuration,
training history, checkpoint, threshold sweep, pixel metrics, object table,
false-positive distribution figure, and per-image review figures. The review
figures use Channel 2 Input only.
