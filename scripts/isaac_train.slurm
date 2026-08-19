#!/bin/bash
#SBATCH --job-name=cilia-unet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge
# Adjust these module names for ISAAC's current software stack.
module load python/3.11
module load cuda

cd /path/to/Cilia-Detector-2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python scripts/prepare_dataset.py \
  --image-dir data/raw \
  --out-dir data/patches \
  --patch-size 256 \
  --stride 128

python scripts/train_unet.py \
  --patch-dir data/patches \
  --out checkpoints/cilia_unet.pt \
  --epochs 80 \
  --batch-size 16

