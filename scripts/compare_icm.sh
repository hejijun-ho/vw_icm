#!/usr/bin/env bash
# Run Standard ICM and VW-ICM side-by-side, then compare.
set -e

PYTHON=/opt/homebrew/Caskroom/miniconda/base/envs/icm/bin/python

echo "=== Training Standard ICM ==="
$PYTHON train.py --config configs/minigrid_standard_icm.yaml

echo ""
echo "=== Training VW-ICM ==="
$PYTHON train.py --config configs/minigrid_vw_icm.yaml

echo ""
echo "=== Comparing ==="
$PYTHON evaluate.py \
  --checkpoint_std runs/standard_icm/final.pt \
  --config_std     configs/minigrid_standard_icm.yaml \
  --checkpoint_vw  runs/vw_icm/final.pt \
  --config_vw      configs/minigrid_vw_icm.yaml \
  --n_episodes 20 \
  --output comparison.png
