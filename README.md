# VW-ICM: Variance-Weighted Intrinsic Curiosity Module

A modular RL exploration framework comparing **Standard ICM** (Pathak et al., 2017) with **VW-ICM**, a variance-weighted extension that improves exploration in environments with locally structured state transitions.

---

## Environment Setup

```bash
conda create -n icm python=3.10 -y
conda activate icm
pip install -r requirements.txt
```

---

## Project Structure

```
2drl_proj/
├── icm/
│   ├── encoder.py          # CNNEncoder / MLPEncoder, make_encoder() factory
│   ├── icm_standard.py     # Standard ICM
│   ├── icm_vw.py           # Variance-Weighted ICM (VW-ICM)
│   └── running_stats.py    # EMAVariance + RunningMeanStd
├── algorithms/
│   ├── base.py             # Abstract base class
│   └── ppo.py              # PPO with ICM plug-in (discrete + continuous)
├── envs/
│   └── wrappers.py         # MiniGridImageWrapper, make_env(), make_vec_env()
├── configs/
│   ├── minigrid_n2_standard_icm.yaml
│   ├── minigrid_n2_vw_icm.yaml
│   ├── minigrid_n4_standard_icm.yaml
│   └── minigrid_n4_vw_icm.yaml
├── train.py                # Training entry point
└── evaluate.py             # Evaluation & comparison
```

---

## Training & Evaluation

```bash
# Train
python train.py --config configs/minigrid_n4_vw_icm.yaml
python train.py --config configs/minigrid_n4_standard_icm.yaml

# Evaluate (compare two checkpoints across multiple seeds)
python evaluate.py \
  --checkpoint_std runs/n4_standard_icm/final.pt \
  --config_std     configs/minigrid_n4_standard_icm.yaml \
  --checkpoint_vw  runs/n4_vw_icm/final.pt \
  --config_vw      configs/minigrid_n4_vw_icm.yaml \
  --n_episodes 100 --eval_seed 100 --output comparison.png
```

---

## Implementation Details

### Notation

| Symbol | Type | Definition |
|--------|------|------------|
| `s`, `s'` | state | current and next environment observation |
| `a` | action | agent action; integer for discrete, vector for continuous |
| `φ(·)` | encoder `ℝ^obs → ℝ^D` | shared feature extractor (CNN or MLP); output is L2-normalized via LayerNorm+Tanh |
| `D` | integer | feature dimension (e.g. 256 for N4) |
| `d` | index | dimension index, `d ∈ {1, …, D}` |
| `B` | integer | batch size |
| `f(·)` | forward model `ℝ^(D+A) → ℝ^D` | predicts `φ(s')` from `[φ(s), action_enc]` |
| `g(·)` | inverse model `ℝ^(2D) → ℝ^A` | predicts action from `[φ(s), φ(s')]` |
| `action_enc` | vector | `one_hot(a)` for discrete; `a` directly for continuous |
| `φ̂(s')` | vector `ℝ^D` | forward model output (predicted next features) |
| `â` | scalar or vector | inverse model output (predicted action) |
| `r_i` | scalar | intrinsic reward for one transition |
| `α` | float | EMA decay rate for variance tracker (default 0.01) |
| `σ²_d` | float | EMA-estimated variance of dimension `d` across transitions |
| `w_d` | float | variance weight for dimension `d`; `w_d ≥ 0` |
| `t` | integer | current training step |

---

### Standard ICM

Standard ICM (Pathak et al., 2017) trains a forward model and an inverse model jointly with the policy. The forward prediction error serves as intrinsic reward, incentivising the agent to visit states the model cannot yet predict.

#### Encoder φ

```
φ(s)  ∈ ℝ^D    — output of CNN (for image obs) or MLP (for flat obs)
φ(s') ∈ ℝ^D    — computed from next observation s'
```

The encoder is **shared** with the actor-critic. It is not updated by the forward model loss; it is trained only through the inverse model and the PPO policy loss.

#### Forward Model f

```
input  : [φ(s), action_enc]  ∈ ℝ^(D+A)
output : φ̂(s')               ∈ ℝ^D       — predicted features of s'
target : φ(s').detach()                   — encoder output, stop-gradient

per-dim error  : e_d = (φ̂(s')_d − φ(s')_d)²          ∈ ℝ^D, shape (B, D)
forward loss   : L_fwd = 0.5 · mean_{b,d}(e_d)         scalar
intrinsic reward: r_i  = 0.5 · mean_d(e_d)             ∈ ℝ,  shape (B,)
```

`r_i` is the average per-dimension squared prediction error for one transition. Higher error = more novel transition = higher intrinsic reward.

#### Inverse Model g

```
input  : [φ(s), φ(s')]  ∈ ℝ^(2D)
output : â

L_inv = CrossEntropy(â, a)   — discrete action space
      = MSE(â, a)            — continuous action space
```

The inverse model forces the encoder to retain only the information needed to predict the agent's own action — filtering out environment features that actions cannot change (distractors, background).

#### Total ICM Loss

```
L_icm = λ_fwd · L_fwd  +  λ_inv · L_inv
      = 0.2   · L_fwd  +  0.8   · L_inv
```

`L_icm` is added directly to the PPO loss and optimised in the same backward pass.

---

### VW-ICM

VW-ICM extends Standard ICM by reweighting the per-dimension forward prediction error with per-dimension variance estimates. The motivation: in locally structured environments, each action affects only a subset of feature dimensions. Upweighting those "active" dimensions concentrates the intrinsic reward signal; downweighting inactive dimensions suppresses noise.

#### Forward Model and Encoder

Identical to Standard ICM. `f` predicts `φ(s')` directly; the encoder is trained the same way via the inverse model.

#### Feature Delta and Variance Tracking

```
δ_d  =  φ(s')_d − φ(s)_d     ∈ ℝ,  per-dimension, per-transition

δ captures how much feature dimension d changed due to action a.
Dimensions with consistently large δ are "active" (sensitive to actions).
Dimensions with δ ≈ 0 carry no action-relevant information.
```

The per-dimension variance of δ is tracked with an EMA:

```
σ²_d  ←  (1 − α) · σ²_d  +  α · Var_b(δ_d)

where Var_b(δ_d) = variance of δ_d across the B samples in the current batch.
α = 0.01 by default; update is skipped when B = 1 (single-step calls
contribute zero batch variance and would only decay σ² without adding signal).
```

Initial value: `σ²_d = 1` for all `d`, so early weights start near uniform.

#### Variance Weights w

```
w_d  =  clamp( σ²_d / mean_d(σ²),  min=0,  max=5.0 )

Normalisation: dividing by mean_d(σ²) ensures mean_d(w_d) ≈ 1 when no
clamping occurs — i.e. the total scale of intrinsic reward matches
Standard ICM in the isotropic case.

Cap at 5.0: prevents any single dimension from contributing more than 5×
the average, guarding against a minority of high-variance dimensions
dominating the entire reward signal.

Warmup: for the first `warmup_steps` variance-tracker updates, w_d = 1
for all d (uniform), equivalent to Standard ICM. This avoids noisy
weights from insufficient early statistics.
```

Behaviour summary:

| Condition | σ²_d | w_d | Effect |
|-----------|------|-----|--------|
| Dimension changes a lot with actions | high | > 1 | amplified in intrinsic reward |
| Dimension is stable regardless of actions | ≈ 0 | ≈ 0 | suppressed |
| All dimensions change equally (isotropic) | uniform | = 1 | reduces to Standard ICM |

#### Intrinsic Reward (variance-weighted)

```
per-dim error  : e_d    = (φ̂(s')_d − φ(s')_d)²      shape (B, D)
weighted error : ẽ_d    = w_d · e_d                   shape (B, D)  — w broadcast over B
intrinsic reward: r_i   = 0.5 · mean_d(ẽ_d)           shape (B,)
```

#### ICM Training Loss (unweighted)

```
L_fwd  = 0.5 · mean_{b,d}(e_d)    — same as Standard ICM, w_d NOT applied
L_icm  = 0.2 · L_fwd  +  0.8 · L_inv
```

**Why keep the training loss unweighted?** Applying `w_d` to `L_fwd` would change which dimensions the encoder is trained to predict accurately, creating a feedback loop: high-variance dims get upweighted → encoder focuses on them → their variance changes → weights shift. Keeping `L_fwd` unweighted means the encoder trains identically to Standard ICM; only the exploration signal (reward) changes.

#### Non-Degradation Guarantee

If all feature dimensions have equal variance (`σ²_d = c` for all `d`), then `w_d = c / c = 1` for all `d`, and `r_i` reduces to exactly the Standard ICM intrinsic reward. VW-ICM cannot perform worse than Standard ICM in isotropic environments.

---

### Standard ICM vs VW-ICM: Differences

| Component | Standard ICM | VW-ICM |
|-----------|-------------|--------|
| Encoder φ | shared with actor-critic | shared with actor-critic (same) |
| Forward model target | `φ(s')` | `φ(s')` (same) |
| Per-dim error `e_d` | `(φ̂(s')_d − φ(s')_d)²` | `(φ̂(s')_d − φ(s')_d)²` (same) |
| Intrinsic reward `r_i` | `0.5 · mean_d(e_d)` | `0.5 · mean_d(w_d · e_d)` |
| ICM training loss `L_fwd` | `0.5 · mean(e_d)` | `0.5 · mean(e_d)` (same, unweighted) |
| Variance tracker | none | `σ²_d` per dim, EMA-updated each batch |
| Weights `w_d` | implicit 1 everywhere | `clamp(σ²_d / mean(σ²), 0, 5)` |
| Warmup behaviour | — | `w_d = 1` for first `warmup_steps` updates |
| Isotropic environments | baseline | reduces to Standard ICM |
| Inverse model / encoder training | unchanged | unchanged (same) |

**In short**: VW-ICM adds one component — the EMA variance tracker — and changes one quantity — how `r_i` is computed from `e_d`. Everything else is identical.

---

### EMAVariance (`icm/running_stats.py`)

**Purpose**: Track the per-dimension variance of feature deltas `δ_d = φ(s')_d − φ(s)_d` across training, and produce normalised weights `w_d` for VW-ICM.

**State variables**:

| Variable | Shape | Initial value | Meaning |
|----------|-------|--------------|---------|
| `ema_var` | `(D,)` | `1.0` for all d | EMA of per-dim variance of δ |
| `step` | scalar | `0` | number of variance-tracker updates received |

**Update rule** (called once per PPO mini-batch, skipped for B=1):

```
Var_b(δ_d)  =  variance of δ_d across the B transitions in the batch
ema_var_d   ←  (1 − α) · ema_var_d  +  α · Var_b(δ_d)
step        ←  step + 1
```

**Weight computation** (called when assembling intrinsic reward):

```
if step < warmup_steps:
    w_d = 1.0  for all d          — uniform, no reweighting

else:
    w_d = clamp( ema_var_d / mean_d(ema_var),  min=0,  max=5.0 )
```

**Behaviour**:
- `α = 0.01`: very slow EMA. After 100 updates the tracker has seen roughly 63% of its converged value. This avoids reacting to transient variance spikes.
- `warmup_steps = 2000`: initial statistics are noisy (random encoder). During warmup, `w_d = 1` prevents premature reweighting from distorting early exploration.
- `max = 5.0`: hard cap. Without it, a renormalisation step after clamping can push weights back above the intended ceiling (see Implementation Notes §1).
- `B = 1` skip: during rollout collection, ICM is called one step at a time (`B=1`). `Var_b(δ_d)` of a single sample is identically zero, so such calls would only decay `ema_var` without contributing signal. Skipping them preserves the ratio of signal to decay.

---

### RunningMeanStd (`icm/running_stats.py`)

**Purpose**: Normalise the scalar intrinsic reward `r_i` so its scale stays stable throughout training, even as the forward model improves and raw `r_i` changes by orders of magnitude.

**State variables**:

| Variable | Initial value | Meaning |
|----------|--------------|---------|
| `var` | `1.0` | EMA of `r²` (approximates `E[r²] ≈ Var(r)` assuming zero mean) |

**Update rule** (called once per environment step with the raw scalar `r_i`):

```
var  ←  (1 − β) · var  +  β · r_i²        β = 0.005
std  =  max( sqrt(var + ε),  min_std )     ε = 1e-8,  min_std = 0.05
r_normalised  =  clip( r_i / std,  0,  5.0 )
```

**Behaviour**:
- `β = 0.005`: slower than the variance tracker (α=0.01). The normaliser adapts gradually so a sudden drop in `r_i` doesn't immediately amplify noise.
- `min_std = 0.05` floor: when the forward model has fully overfit, `r_i → 0` and `var → 0`. Without a floor, `std → 0` and even tiny residual noise would be normalised to the clip ceiling (5.0), producing a spuriously large signal. The floor keeps the normalised reward small when the raw reward is small.
- `initial var = 1.0`: treats the first step as if `std ≈ 1`, preventing large early rewards from destabilising the policy before the EMA has converged.
- `clip(·, 0, 5.0)`: intrinsic reward is non-negative by construction (squared error), and the upper clip prevents single-step outliers from dominating the advantage estimate.

---

### Intrinsic Coefficient Decay

**Purpose**: Anneal the weight of intrinsic reward to zero by the end of training, so the saved policy operates under the same conditions as evaluation (no intrinsic reward).

**Schedule** (linear decay starting at fraction `T_decay` of total steps):

```
let  T_start = T_decay · total_steps       (default T_decay = 0.8)

         ⎧  intrinsic_coeff                           t < T_start
coeff(t) = ⎨
         ⎩  intrinsic_coeff · max(0, 1 − (t − T_start) / (total_steps − T_start))   t ≥ T_start
```

**Behaviour**:
- For the first 80% of training (`t < T_start`): full intrinsic coefficient. The agent explores freely.
- From 80% to 100%: coefficient linearly decreases from `intrinsic_coeff` to 0. The agent transitions from exploration-driven to purely extrinsic behaviour.
- At `t = total_steps`: `coeff = 0`. The final checkpoint is saved after at least one full decay cycle, so evaluation with no intrinsic reward is consistent with how the policy was last trained.

Without decay: the EMA normaliser keeps intrinsic reward non-zero even after the forward model has converged (due to the `min_std` floor). The policy therefore never learns to navigate on extrinsic reward alone, and fails at evaluation.

---

## Config Reference

```yaml
env: MiniGrid-MultiRoom-N4-S5-v1
action_space: discrete          # discrete | continuous
total_steps: 5000000
max_steps: 2000                 # per-episode step limit (overrides env default)
seed: 42
n_envs: 8                       # parallel environments via AsyncVectorEnv

encoder:
  type: auto                    # auto | cnn | mlp
  feature_dim: 256

icm:
  type: vw                      # vw | standard
  intrinsic_coeff: 0.01
  normalize_intrinsic: true
  intrinsic_decay_start: 0.8    # fraction of total_steps when decay begins
  forward_loss_coeff: 0.2
  inverse_loss_coeff: 0.8
  # VW-ICM only:
  ema_alpha: 0.01
  warmup_steps: 2000

ppo:
  lr: 2.5e-4
  n_steps: 256                  # steps per env per rollout
  n_epochs: 4
  batch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coeff: 0.5
  ent_coeff: 0.01

logging:
  log_dir: runs/n4_vw_icm
  save_freq: 500000
  log_freq: 20
```

---

## Implementation Notes

Issues discovered and fixed during development, ordered roughly by impact:

**1. Variance weight cap broken by renormalization**

`EMAVariance.weights` first clamped raw weights to `max=5.0`, then divided by `raw.mean()`. If most dimensions had near-zero variance, `mean < 1`, so the clamped-at-5 values were pushed back above 5 (observed `weight_max = 7.8` in logs). Fix: apply the cap *after* normalization (`ema_var / mean(ema_var)` already has mean=1, so clamp directly).

**2. Intrinsic coefficient decay required**

Without decay, the intrinsic reward remains active at the end of training (EMA normalization keeps a non-zero floor). The policy learns to navigate *with* intrinsic reward but fails at evaluation *without* it. Adding a linear decay over the last 20% of training forces the policy to learn pure exploitation before the checkpoint is saved.

**3. Environment reset between rollouts**

The original `collect_rollout` called `env.reset()` at the start of every rollout. This capped effective episode length at `n_steps` steps — any episode longer than one rollout was silently truncated and abandoned. The fix stores `_last_obs` on the PPO object and only resets on the very first call, allowing episodes to continue across rollout boundaries.

**4. Variance tracker polluted by single-step rollout calls**

During rollout collection, `icm.forward()` was called with `batch_size=1`. For a single sample, `delta.var(dim=0, unbiased=False) = 0`, so every rollout step only decayed `ema_var` without adding signal. With 256 rollout steps per update vs 4 PPO batch calls, the tracker saw 64× more decay than signal. Fix: skip the variance tracker update when `batch_size == 1`.

**5. Gradient clipping covered only actor-critic**

`nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)` missed the ICM forward and inverse model parameters. These share no parameters with the actor-critic (only the encoder is shared, and it is included in `ac.parameters()`). Fix: store `self._all_params` at optimizer construction time and clip that list.

**6. Checkpoint save condition broke with vectorized envs**

With `n_envs=8`, `steps_done` increments by `n_steps * n_envs = 2048` per update. The save condition `steps_done % save_freq < n_steps` (where `n_steps=256`) would never trigger because the step jump crosses the boundary with a remainder of ~736 > 256. Fix: change to `< steps_per_update`.

**7. Forward model target: φ(s') not δ**

An earlier design had the VW-ICM forward model predict the feature *delta* δ = φ(s') - φ(s). For a randomly initialized encoder, δ ≈ 0 everywhere, giving near-zero training signal from the start. Predicting φ(s') directly (same as Standard ICM) gives a non-trivial target and stable training from step 1.
