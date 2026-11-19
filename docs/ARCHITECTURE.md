# Architecture

## Design Principle

The project performs binary semantic segmentation: cilia are `1` and background is `0`. Every U-Net workflow selects only zero-based `CHANNEL_INDEX = 1`, which is the second microscopy channel. This rule is enforced in core preprocessing and checkpoint loading rather than only in the user interface.

## Code Map

| Location | Responsibility |
| --- | --- |
| `scripts/cilia_app.py` | Main local FastAPI browser app. Owns browser routes, background training jobs, report generation, and the embedded single-page interface. |
| `scripts/serve_channel2_api.py` | Small standalone API intended for Colab or programmatic use. |
| `cilia_detector/channel2_pipeline.py` | Shared array/file preprocessing, image-mask pairing, one-channel U-Net training, checkpoint I/O, and prediction helpers. |
| `cilia_detector/io.py` | TIFF stack adapters, including ImageJ/OME-style channel and multi-image prediction stacks. |
| `cilia_detector/denoise.py` | Selected-channel normalization and denoising helpers for full-resolution app workflows. |
| `cilia_detector/model.py` | The U-Net architecture and Dice loss. |
| `cilia_detector/morphology.py` | Connected-object morphology measurements used by mask analysis, prediction, and evaluation. |
| `cilia_detector/dataset.py` | Patch serialization used by the app and legacy patch-based CLI trainer. |
| `scripts/` | Optional command-line prediction, training, optimization, classification, denoising, and cluster utilities. |
| `notebooks/` | Self-contained Colab workflow. |
| `tests/` | Regression tests for data handling and recovery behavior. |

## Main Browser-App Flow

```text
Uploaded image/mask ZIP or image file
        |
        v
TIFF/file adapters (`io.py`, `channel2_pipeline.py`)
        |
        v
Select only image[..., 1:2] and normalize that channel
        |
        +--> Train / continue training --> U-Net checkpoint
        |
        +--> Predict --> binary mask --> morphology and t-SNE outputs
        |
        +--> Test --> metrics, plots, and failure-review figures
```

`scripts/cilia_app.py` is intentionally a single module because its browser HTML, route handlers, background-job state, and report-generation helpers share the same local-filesystem context. Reusable image/model operations belong in `cilia_detector/` instead of being duplicated in new routes or scripts.

## Where To Make Changes

| Goal | Start Here |
| --- | --- |
| Change the U-Net layers or loss | `cilia_detector/model.py` |
| Change the selected input channel or normalized image preprocessing | `cilia_detector/channel2_pipeline.py` and relevant tests |
| Support another TIFF axis arrangement | `cilia_detector/io.py` plus a regression test |
| Change metrics, failure reviews, or reports | `scripts/cilia_app.py` |
| Change mask shape measurements/t-SNE input features | `cilia_detector/morphology.py` and app analysis helpers |
| Change browser layout or labels | Embedded HTML/CSS/JavaScript in `scripts/cilia_app.py` |
| Add an automation-friendly endpoint | `scripts/serve_channel2_api.py` |

## Persistence and Generated Files

- `checkpoints/` stores local `.pt` model checkpoints.
- `outputs/` stores generated masks, plots, CSV files, and report images.
- `.build/` is temporary test/runtime output.
- These paths are ignored by Git. Source code, documentation, small test fixtures, and configuration should be committed; microscopy data, model weights, and generated analyses should not.

## Compatibility Guardrails

- Checkpoints must declare one model input channel and `channel_index = 1`.
- The app rejects unsupported or ambiguous image layouts with an actionable message.
- Training and testing pair files by stem and fail before work begins if masks are missing.
- Background training is intentionally serialized in the local app to avoid competing for laptop memory.
