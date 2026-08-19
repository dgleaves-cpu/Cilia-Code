# Google Colab Channel-2 U-Net Workflow

Use the complete notebook here:

[notebooks/channel2_cilia_unet_colab.ipynb](notebooks/channel2_cilia_unet_colab.ipynb)

The notebook is configured for the revised rule: the model uses only zero-based `CHANNEL_INDEX = 1`, meaning the second image channel. After standardizing each image to `(height, width, channels)`, it selects `image[..., 1:2]`, preserving the channel dimension as `(height, width, 1)`.

## Expected Folders

```text
MyDrive/cilia-detector-2/
  data/
    images/
      CS02-001.tif
      CS02-002.tif
    masks/
      CS02-001.tif
      CS02-002.tif
  checkpoints/
```

Image and mask files are matched by filename stem. For example, `images/CS02-001.tif` matches `masks/CS02-001.tif`.

Regenerate any cached training patches after this update. Older patch folders from the previous workflow may contain two input channels.

## Notebook Features

- Configuration variables: `IMAGE_DIR`, `MASK_DIR`, `IMAGE_SIZE`, `CHANNEL_INDEX = 1`, and `MODEL_PATH`.
- Paired image/mask loader with helpful missing-mask errors.
- One-channel PyTorch U-Net with sigmoid binary segmentation output.
- BCE + Dice loss, Dice coefficient, and IoU reporting.
- Visualizations limited to `Channel 2 Input`, `True Cilia Mask`, and `Predicted Cilia Mask`.
- FastAPI service for `/health`, `/predict`, and `/train` through ngrok.

Retraining or fine-tuning requires images and matching manually prepared cilia masks. Prediction requires only new multichannel images.
