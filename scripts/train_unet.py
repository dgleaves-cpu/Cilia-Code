#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cilia_detector.model import UNet, dice_loss


class PatchDataset(Dataset):
    def __init__(self, patch_dir: Path):
        self.paths = sorted(patch_dir.glob("*.npz"))
        if not self.paths:
            raise ValueError(f"No .npz patches found in {patch_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        data = np.load(self.paths[idx])
        image = torch.from_numpy(data["image"]).float()
        mask = torch.from_numpy(data["mask"][None, ...]).float()
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        return image, mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small U-Net for cilium segmentation.")
    parser.add_argument("--patch-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = PatchDataset(args.patch_dir)
    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet(in_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cilia are tiny relative to the whole field of view, so most pixels are
    # background. The positive class weight makes missing cilium pixels more
    # expensive, while Dice loss rewards overlap of thin cilium structures.
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0], device=device))

    best_val = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                val_loss += float(loss.item()) * images.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_channels": 1,
                    "threshold": 0.5,
                    "best_val_loss": best_val,
                },
                args.out,
            )
            print(f"saved {args.out}")


if __name__ == "__main__":
    main()
