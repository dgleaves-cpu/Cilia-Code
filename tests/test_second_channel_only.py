from __future__ import annotations

import asyncio
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import tifffile
from PIL import Image

from cilia_detector.channel2_pipeline import (
    CHANNEL_INDEX,
    load_unet_model,
    png_bytes_from_mask,
    preprocess_image_array,
    select_channel_preserve_dim,
)
from cilia_detector.model import UNet
from cilia_detector.io import read_prediction_stacks, read_stack
from scripts import cilia_app
from scripts.cilia_app import api_analyze_masks


class SecondChannelOnlyTests(unittest.TestCase):
    def test_selects_only_zero_based_channel_one_with_dimension_preserved(self) -> None:
        image = np.zeros((20, 21, 5), dtype=np.float32)
        image[..., CHANNEL_INDEX] = 7.0
        image[..., 0] = 99.0
        image[..., 2:] = -5.0

        selected = select_channel_preserve_dim(image)

        self.assertEqual(selected.shape, (20, 21, 1))
        np.testing.assert_array_equal(selected[..., 0], image[..., CHANNEL_INDEX])

    def test_preprocessing_is_unchanged_when_other_channels_change(self) -> None:
        image = np.zeros((20, 21, 5), dtype=np.float32)
        image[..., CHANNEL_INDEX] = np.arange(20 * 21, dtype=np.float32).reshape(20, 21)
        changed_other_channels = image.copy()
        changed_other_channels[..., 0] = 1_000_000.0
        changed_other_channels[..., 2:] = -1_000_000.0

        baseline = preprocess_image_array(image, image_size=(16, 16))
        changed = preprocess_image_array(changed_other_channels, image_size=(16, 16))

        self.assertEqual(baseline.shape, (16, 16, 1))
        np.testing.assert_allclose(baseline, changed)

    def test_checkpoint_with_wrong_channel_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_channel.pt"
            torch.save(
                {
                    "model_state": UNet(in_channels=1).state_dict(),
                    "in_channels": 1,
                    "channel_index": 0,
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "second image channel"):
                load_unet_model(path, device=torch.device("cpu"))

    def test_mask_png_uses_red_cilia_on_a_white_background(self) -> None:
        mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        image = np.asarray(Image.open(BytesIO(png_bytes_from_mask(mask))).convert("RGB"))

        np.testing.assert_array_equal(image[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(image[0, 1], [214, 39, 40])

    def test_reads_imagej_i_axis_stacks_as_channel_first(self) -> None:
        stack = np.arange(5 * 12 * 13, dtype=np.uint16).reshape(5, 12, 13)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "imagej_i_axis.tif"
            tifffile.imwrite(path, stack, metadata={"axes": "IYX"})
            loaded, _ = read_stack(path)

        self.assertEqual(loaded.shape, (5, 12, 13))
        np.testing.assert_array_equal(loaded, stack)

    def test_splits_multimage_tiff_into_channel_first_prediction_stacks(self) -> None:
        images = np.zeros((2, 5, 12, 13), dtype=np.uint16)
        images[0, CHANNEL_INDEX] = 11
        images[1, CHANNEL_INDEX] = 22

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "time_series.tif"
            tifffile.imwrite(path, images, metadata={"axes": "TCYX"})
            stacks = read_prediction_stacks(path)

        self.assertEqual([stack.shape for stack in stacks], [(5, 12, 13), (5, 12, 13)])
        self.assertEqual(stacks[0][CHANNEL_INDEX, 0, 0], 11)
        self.assertEqual(stacks[1][CHANNEL_INDEX, 0, 0], 22)

    def test_batch_prediction_exports_one_mask_per_image_and_manifest(self) -> None:
        images = np.zeros((2, 5, 32, 32), dtype=np.uint16)
        images[:, CHANNEL_INDEX, 8:16, 8:10] = 100
        upload_bytes = BytesIO()
        tifffile.imwrite(upload_bytes, images, metadata={"axes": "TCYX"})
        upload_bytes.seek(0)

        def fake_prediction(*_: object, **__: object):
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:16, 8:10] = 1
            channel = np.zeros((32, 32, 1), dtype=np.float32)
            objects = pd.DataFrame(
                [{"object_id": 1, "area_px": 16, "length_px": 8.0, "skeleton_path_length_px": 7.0,
                  "endpoint_distance_px": 7.0, "curvature_tortuosity": 1.0, "centroid_y": 11.5, "centroid_x": 8.5}]
            )
            return np.zeros((32, 32), dtype=np.float32), mask, channel, objects

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.pt"
            model_path.touch()
            output_dir = root / "predict"
            upload = SimpleNamespace(filename="time_series.tif", file=upload_bytes)
            with (
                patch.object(cilia_app, "ensure_output_dir", return_value=output_dir),
                patch.object(cilia_app, "relative_url", side_effect=lambda path: f"/outputs/{Path(path).name}"),
                patch.object(cilia_app, "load_unet_model", return_value=(object(), {"image_size": (32, 32)})),
                patch.object(cilia_app, "full_resolution_prediction_from_stack", side_effect=fake_prediction),
            ):
                payload = asyncio.run(
                    cilia_app.api_predict(
                        images=[upload],
                        image=None,
                        model_path=str(model_path),
                        threshold=0.5,
                        full_resolution=True,
                        tile_size=64,
                        overlap=0,
                        min_object_size=1,
                    )
                )

            with ZipFile(output_dir / "predicted_cilia_masks.zip") as archive:
                names = sorted(archive.namelist())

        self.assertEqual(payload["processed_images"], 2)
        self.assertEqual(len(payload["prediction_items"]), 2)
        self.assertEqual(
            names,
            [
                "masks/time_series_image_001_predicted_cilia_mask.tif",
                "masks/time_series_image_002_predicted_cilia_mask.tif",
                "predicted_mask_manifest.csv",
            ],
        )

    def test_mask_analysis_accepts_a_mask_only_zip_and_returns_tsne_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zip_path = Path(directory) / "masks.zip"
            with ZipFile(zip_path, "w") as archive:
                for index in range(4):
                    mask = np.zeros((48, 48), dtype=np.uint8)
                    mask[5 + index : 12 + index, 8 : 10] = 255
                    mask[24 : 27, 20 + index : 31 + index] = 255
                    image = Image.fromarray(mask)
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    archive.writestr(f"masks/sample_{index:03d}.png", buffer.getvalue())
                # Finder-created ZIPs often include AppleDouble records with image-like filenames.
                archive.writestr("__MACOSX/masks/._sample_000.png", b"not an image")

            with zip_path.open("rb") as handle:
                payload = asyncio.run(
                    api_analyze_masks(
                        SimpleNamespace(filename="masks.zip", file=handle),
                        min_object_size=1,
                    )
                )

        self.assertEqual(payload["analyzed_masks"], 4)
        self.assertGreaterEqual(payload["measured_cilia"], 4)
        self.assertIn("morphology_csv_url", payload)
        self.assertIn("tsne_length_plot_url", payload)
        self.assertIn("tsne_curvature_plot_url", payload)

if __name__ == "__main__":
    unittest.main()
