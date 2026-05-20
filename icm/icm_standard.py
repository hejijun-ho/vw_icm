import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class StandardICM(nn.Module):
    """Standard Intrinsic Curiosity Module (Pathak et al., 2017).

    Intrinsic reward = forward model prediction error in feature space.
    The encoder is trained via the inverse model (predicting action from
    consecutive state features), learning task-relevant representations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        is_discrete: bool = True,
        forward_loss_coeff: float = 0.2,
        inverse_loss_coeff: float = 0.8,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.forward_loss_coeff = forward_loss_coeff
        self.inverse_loss_coeff = inverse_loss_coeff

        feat = encoder.feature_dim

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

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.is_discrete:
            return F.one_hot(action.long(), self.action_dim).float()
        return action.float()

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute intrinsic reward and ICM losses.

        Args:
            obs:      (B, *obs_shape)
            action:   (B,) for discrete or (B, action_dim) for continuous
            next_obs: (B, *obs_shape)

        Returns:
            intrinsic_reward: (B,) per-sample reward
            info: dict with 'forward_loss', 'inverse_loss', 'icm_loss'
        """
        phi_s = self.encoder(obs)
        phi_s_next = self.encoder(next_obs)
        action_enc = self._encode_action(action)

        # Forward model: predict φ(s') from φ(s) and action
        phi_s_pred = self.forward_model(torch.cat([phi_s, action_enc], dim=-1))
        forward_loss_vec = 0.5 * (phi_s_pred - phi_s_next.detach()).pow(2).mean(dim=-1)
        forward_loss = forward_loss_vec.mean()

        # Intrinsic reward = per-sample forward prediction error
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
        }
        return intrinsic_reward, icm_loss, info
