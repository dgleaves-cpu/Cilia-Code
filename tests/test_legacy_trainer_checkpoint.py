from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cilia_detector.channel2_pipeline import CHANNEL_INDEX
from scripts import train_unet


class LegacyTrainerCheckpointTests(unittest.TestCase):
    def test_saved_checkpoint_records_channel2_and_patch_size(self) -> None:
        class TinyUNet(torch.nn.Module):
            def __init__(self, in_channels: int) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(1))
                self.in_channels = in_channels

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.bias.expand_as(images[:, :1])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch_dir = root / "patches"
            patch_dir.mkdir()
            for index in range(2):
                np.savez_compressed(
                    patch_dir / f"patch_{index}.npz",
                    image=np.zeros((1, 16, 20), dtype=np.float32),
                    mask=np.zeros((16, 20), dtype=np.float32),
                    source="sample",
                )
            checkpoint_path = root / "checkpoint.pt"

            arguments = [
                "train_unet.py",
                "--patch-dir",
                str(patch_dir),
                "--out",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--num-workers",
                "0",
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(train_unet, "UNet", TinyUNet),
                patch.object(train_unet, "tqdm", side_effect=lambda items, **_: items),
            ):
                train_unet.main()

            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.assertEqual(checkpoint["in_channels"], 1)
        self.assertEqual(checkpoint["channel_index"], CHANNEL_INDEX)
        self.assertEqual(checkpoint["image_size"], (16, 20))


if __name__ == "__main__":
    unittest.main()
