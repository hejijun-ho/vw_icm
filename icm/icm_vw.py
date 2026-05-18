import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .running_stats import EMAVariance


class VWICM(nn.Module):
    """Variance-Weighted Intrinsic Curiosity Module (VW-ICM).

    Extends standard ICM by reweighting the forward prediction error
    per feature dimension based on observed EMA variance of δ = φ(s') - φ(s).

    Key property: In isotropic environments (all dimensions change equally),
    weights collapse to uniform → reduces exactly to standard ICM.
    In anisotropic environments (locally structured transitions), weights
    amplify signal from active dimensions and suppress inactive ones.

    Intrinsic reward:
        r_i = Σ_d  w_d · (f_d(φ(s), a) - φ(s')_d)²
        where w_d = σ²_d / mean(σ²)   (normalized EMA variance, mean=1)
        and   σ²_d = EMA variance of δ_d = φ(s')_d - φ(s)_d
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        is_discrete: bool = True,
        forward_loss_coeff: float = 0.2,
        inverse_loss_coeff: float = 0.8,
        ema_alpha: float = 0.01,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.forward_loss_coeff = forward_loss_coeff
        self.inverse_loss_coeff = inverse_loss_coeff

        feat = encoder.feature_dim

        # Forward model predicts φ(s') directly; δ variance tracked separately for weights
        self.forward_model = nn.Sequential(
            nn.Linear(feat + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, feat),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(feat * 2, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
        )

        self.variance_tracker = EMAVariance(feat, alpha=ema_alpha, warmup_steps=warmup_steps)

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.is_discrete:
            return F.one_hot(action.long(), self.action_dim).float()
        return action.float()

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute variance-weighted intrinsic reward and ICM losses.

        Key design: forward model predicts φ(s') directly (same as standard ICM,
        ensures non-trivial prediction target from the start). Variance of
        δ = φ(s') - φ(s) is tracked separately and used to reweight the
        per-dimension prediction error — amplifying dimensions that actually
        change with actions, suppressing inactive ones.

        Args:
            obs:      (B, *obs_shape)
            action:   (B,) for discrete or (B, action_dim) for continuous
            next_obs: (B, *obs_shape)

        Returns:
            intrinsic_reward: (B,) per-sample reward
            icm_loss:         scalar loss for optimizer
            info:             logging dict
        """
        phi_s = self.encoder(obs)
        phi_s_next = self.encoder(next_obs)
        action_enc = self._encode_action(action)

        # Track variance of δ to identify "active" feature dimensions
        delta = (phi_s_next - phi_s).detach()
        self.variance_tracker.update(delta)
        weights = self.variance_tracker.weights.to(obs.device)  # (D,)

        # Forward model: predict φ(s') directly (same target as standard ICM)
        phi_next_pred = self.forward_model(torch.cat([phi_s, action_enc], dim=-1))

        error = (phi_next_pred - phi_s_next.detach()).pow(2)      # (B, D)

        # ICM training loss: unweighted — encoder trains identically to standard ICM
        forward_loss = 0.5 * error.mean()

        # Intrinsic reward: variance-weighted — focuses exploration signal on active dims
        weighted_error = weights.unsqueeze(0) * error             # (B, D)
        forward_loss_vec = 0.5 * weighted_error.mean(dim=-1)      # (B,)

        # Intrinsic reward = per-sample weighted prediction error
        intrinsic_reward = forward_loss_vec.detach()

        # Inverse model: predict action from φ(s) and φ(s')
        action_pred = self.inverse_model(torch.cat([phi_s, phi_s_next], dim=-1))
        if self.is_discrete:
            inverse_loss = F.cross_entropy(action_pred, action.long())
        else:
            inverse_loss = F.mse_loss(action_pred, action.float())

        icm_loss = (
            self.forward_loss_coeff * forward_loss
            + self.inverse_loss_coeff * inverse_loss
        )

        info = {
            "forward_loss": forward_loss.item(),
            "inverse_loss": inverse_loss.item(),
            "icm_loss": icm_loss.item(),
            "intrinsic_reward_mean": intrinsic_reward.mean().item(),
            "weight_anisotropy": self.variance_tracker.anisotropy,
            "variance_tracker_step": self.variance_tracker.step.item(),
        }
        return intrinsic_reward, icm_loss, info
