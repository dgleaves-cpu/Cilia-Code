from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import cilia_app


class TrainingRecoveryTests(unittest.TestCase):
    def test_mps_broken_pipe_retries_once_on_cpu_before_a_checkpoint_exists(self) -> None:
        calls: list[str] = []
        recoveries: list[bool] = []

        def fake_train(*_: object, device: torch.device, **__: object):
            calls.append(device.type)
            if device.type == "mps":
                raise BrokenPipeError(32, "Broken pipe")
            return [{"epoch": 1.0, "train_loss": 1.0}]

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(cilia_app, "get_device", return_value=torch.device("mps")),
                patch.object(cilia_app, "train_from_patch_dir", side_effect=fake_train),
            ):
                history = cilia_app.train_with_device_recovery(
                    patch_dir=Path(directory),
                    model_path=Path(directory) / "new_model.pt",
                    epochs=1,
                    batch_size=1,
                    learning_rate=1e-4,
                    val_fraction=0.2,
                    overwrite_model=False,
                    epoch_callback=None,
                    recovery_callback=lambda: recoveries.append(True),
                )

        self.assertEqual(calls, ["mps", "cpu"])
        self.assertEqual(recoveries, [True])
        self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
