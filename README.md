# VW-ICM: Variance-Weighted Intrinsic Curiosity Module

A modular RL exploration framework that extends ICM with **automatic variance-weighted delta prediction**, improving exploration efficiency in environments with locally structured state transitions.

## Core Idea

Standard ICM computes intrinsic reward as the forward prediction error over the entire feature vector. In environments where each action only affects a subset of feature dimensions, the signal is diluted by unchanged dimensions.

**VW-ICM** tracks the per-dimension EMA variance of `δ = φ(s_{t+1}) - φ(s_t)` and reweights the prediction error accordingly:

```
r_i = Σ_d  w_d · (f_d(φ(s_t), a_t) - δ_d)²
where  w_d = σ²_d / mean(σ²)   (normalized EMA variance)
```

This is **automatically adaptive**: in isotropic environments (all dimensions change equally), weights collapse to uniform → reduces to standard ICM.

## Environment Setup

```bash
# Create and activate conda environment
conda create -n icm python=3.10 -y
conda activate icm
pip install -r requirements.txt
```

## Project Structure

```
2drl_proj/
├── icm/
│   ├── encoder.py          # CNN / MLP feature encoder φ
│   ├── icm_standard.py     # Standard ICM (baseline)
│   ├── icm_vw.py           # Variance-Weighted ICM (VW-ICM)
│   └── running_stats.py    # EMA variance tracker
├── algorithms/
│   ├── base.py             # Abstract RL algorithm interface
│   └── ppo.py              # PPO (discrete + continuous action spaces)
├── envs/
│   └── wrappers.py         # Observation wrappers (MiniGrid, etc.)
├── configs/
│   ├── minigrid_standard_icm.yaml
│   └── minigrid_vw_icm.yaml
├── train.py                # Training entry point
├── evaluate.py             # Comparison & evaluation
└── scripts/
    └── compare_icm.sh      # Run standard vs VW-ICM side-by-side
```

## Training

```bash
conda activate icm

# Train with VW-ICM (default)
python train.py --config configs/minigrid_vw_icm.yaml

# Train with standard ICM (baseline)
python train.py --config configs/minigrid_standard_icm.yaml

# Override any config value from CLI
python train.py --config configs/minigrid_vw_icm.yaml --env MiniGrid-MultiRoom-N4-S5-v1 --total_steps 2000000
```

## Compare Standard ICM vs VW-ICM

```bash
# Run both and generate comparison plots
bash scripts/compare_icm.sh

# Or evaluate from existing checkpoints
python evaluate.py \
  --checkpoint_std runs/standard_icm/best.pt \
  --checkpoint_vw  runs/vw_icm/best.pt \
  --env MiniGrid-MultiRoom-N4-S5-v1
```

## Extending to Other Environments

The ICM modules are environment-agnostic. To add a new environment:

1. Add a wrapper in `envs/wrappers.py` if the observation needs preprocessing
2. Create a config in `configs/`
3. For continuous action spaces, set `action_space: continuous` in the config — the algorithm and ICM handle this automatically

## Config Reference

```yaml
env: MiniGrid-MultiRoom-N4-S5-v1
action_space: discrete          # discrete | continuous
total_steps: 2000000
seed: 42

encoder:
  type: cnn                     # cnn | mlp
  feature_dim: 256

icm:
  type: vw                      # vw | standard
  lr: 1e-4
  intrinsic_coeff: 0.01         # weight of intrinsic reward
  ema_alpha: 0.01               # EMA decay for variance tracker (VW-ICM only)
  forward_loss_coeff: 0.2
  inverse_loss_coeff: 0.8

ppo:
  lr: 2.5e-4
  n_steps: 128
  n_epochs: 4
  batch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coeff: 0.5
  ent_coeff: 0.01

logging:
  log_dir: runs/
  save_freq: 100000
```

## Technical Notes

- **Non-degradation guarantee**: In environments where all feature dimensions change uniformly, VW-ICM weights converge to uniform → identical to standard ICM.
- **EMA warm-up**: Variance weights are unreliable in the first ~1000 steps. A warm-up mask suppresses the weighting until sufficient statistics are collected.
- **Encoder sharing**: The encoder φ is shared between the ICM and the policy network by default (configurable).
- **Action space**: PPO supports both discrete (Categorical) and continuous (Gaussian) action distributions. ICM inverse model adapts accordingly.
