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

# This legacy cluster workflow trains from an existing directory of `.npz`
# patches. Create the patches beforehand and set PATCH_DIR to that directory.
# For paired full-resolution images and masks, the local browser app is the
# recommended workflow because it creates patches during training.
PATCH_DIR="${PATCH_DIR:-data/patches}"
if ! compgen -G "${PATCH_DIR}/*.npz" > /dev/null; then
  echo "No .npz patches found in ${PATCH_DIR}. Set PATCH_DIR before submitting."
  exit 1
fi

python scripts/train_unet.py \
  --patch-dir "${PATCH_DIR}" \
  --out checkpoints/cilia_unet.pt \
  --epochs 80 \
  --batch-size 16
