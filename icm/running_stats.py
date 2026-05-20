import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    """EMA-based running std for intrinsic reward normalization.

    Key design:
    - Divides by std only (keeps reward non-negative)
    - min_std floor prevents amplifying residual noise when reward collapses
    - Initialized with var = init_std² so early normalization is stable

    Behavior:
    - Reward active:   raw ≈ 0.2, std ≈ 0.15 → normalized ≈ 1.3  (normal scale)
    - Reward collapse: raw ≈ 0.001, std → 0 but floored at min_std → normalized ≈ 0.01 (stays small)
    """

    def __init__(
        self,
        alpha: float = 0.005,
        min_std: float = 0.05,
        init_std: float = 1.0,
        clip: float = 5.0,
        epsilon: float = 1e-8,
    ):
        self.alpha = alpha
        self.min_std = min_std
        self.clip = clip
        self.epsilon = epsilon
        self.var = init_std ** 2

    def update(self, x: float) -> None:
        self.var = (1 - self.alpha) * self.var + self.alpha * (x ** 2)

    @property
    def std(self) -> float:
        return max((self.var + self.epsilon) ** 0.5, self.min_std)

    def normalize(self, x: float) -> float:
        return float(np.clip(x / self.std, 0.0, self.clip))


class EMAVariance(nn.Module):
    """Per-dimension EMA variance tracker for feature deltas.

    Tracks σ²_d = EMA of (δ_d)² for each feature dimension d.
    Used by VW-ICM to reweight the forward prediction error.

    Warm-up: weights are suppressed (uniform) for the first `warmup_steps`
    updates to avoid noisy statistics early in training.
    """

    def __init__(self, feature_dim: int, alpha: float = 0.01, warmup_steps: int = 1000):
        super().__init__()
        self.alpha = alpha
        self.warmup_steps = warmup_steps

        self.register_buffer("ema_mean", torch.zeros(feature_dim))
        self.register_buffer("ema_var", torch.ones(feature_dim))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def update(self, delta: torch.Tensor) -> None:
        """Update EMA statistics with a batch of deltas.

        Args:
            delta: (B, D) tensor of feature deltas φ(s') - φ(s)
        """
        batch_mean = delta.mean(dim=0).detach()
        batch_var = delta.var(dim=0, unbiased=False).detach()

        self.ema_mean = (1 - self.alpha) * self.ema_mean + self.alpha * batch_mean
        self.ema_var = (1 - self.alpha) * self.ema_var + self.alpha * batch_var
        self.step += 1

    @property
    def weights(self) -> torch.Tensor:
        """Normalized variance weights, shape (D,).

        Returns uniform weights during warm-up. After warm-up, returns
        σ²_d / mean(σ²), so dimensions that vary more get higher weight.
        """
        if self.step < self.warmup_steps:
            return torch.ones_like(self.ema_var)

        mean_var = self.ema_var.mean().clamp(min=1e-8)
        # Normalize by mean so isotropic case gives all-ones weights.
        # Clamp after (not before) normalizing so the cap is never violated
        # by a subsequent renorm step.
        return (self.ema_var / mean_var).clamp(min=0.0, max=5.0)

    @property
    def anisotropy(self) -> float:
        """Scalar measure of how anisotropic the delta distribution is.

        Returns std(weights) — higher means more structured (good for VW-ICM).
        Near 0 means isotropic (uniform weights, same as standard ICM).
        """
        return self.weights.std().item()
