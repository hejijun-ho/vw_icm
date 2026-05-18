import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .base import BaseAlgorithm
from icm.running_stats import RunningMeanStd


class ActorCritic(nn.Module):
    """Shared-encoder Actor-Critic for discrete or continuous action spaces."""

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        is_discrete: bool = True,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.is_discrete = is_discrete
        feat = encoder.feature_dim

        self.actor_head = nn.Sequential(
            nn.Linear(feat, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(feat, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        if not is_discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        features = self.encoder(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)

        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=logits)
        else:
            std = self.log_std.exp().expand_as(logits)
            dist = torch.distributions.Normal(logits, std)

        return dist, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.encoder(obs)).squeeze(-1)


class PPO(BaseAlgorithm):
    """PPO with optional ICM-based intrinsic reward augmentation.

    Supports discrete and continuous action spaces.
    ICM module is treated as a plug-in: pass any StandardICM or VWICM instance.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        icm: Optional[nn.Module] = None,
        lr: float = 2.5e-4,
        icm_lr: float = 1e-4,
        n_steps: int = 128,
        n_epochs: int = 4,
        batch_size: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coeff: float = 0.5,
        ent_coeff: float = 0.01,
        intrinsic_coeff: float = 0.01,
        normalize_intrinsic: bool = True,
        device: str = "cpu",
    ):
        self.ac = actor_critic.to(device)
        self.icm = icm.to(device) if icm is not None else None
        self.device = device
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coeff = vf_coeff
        self.ent_coeff = ent_coeff
        self.intrinsic_coeff = intrinsic_coeff
        self.normalize_intrinsic = normalize_intrinsic
        self._intr_rms = RunningMeanStd() if normalize_intrinsic else None

        # Collect unique parameters (encoder is shared between ac and icm)
        seen = set()
        params = []
        for p in list(actor_critic.parameters()) + (list(icm.parameters()) if icm is not None else []):
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        self.optimizer = torch.optim.Adam(params, lr=lr)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self, env, icm: Optional[nn.Module] = None, episode_returns: Optional[list] = None
    ) -> Dict[str, Any]:
        icm = icm or self.icm
        obs_list, act_list, rew_list, done_list, val_list, logp_list = [], [], [], [], [], []
        next_obs_list = []
        intr_rewards = []

        obs, _ = env.reset()
        ep_ret = 0.0
        for _ in range(self.n_steps):
            obs_t = self._to_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist, value = self.ac(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action)
                if not self.ac.is_discrete:
                    logp = logp.sum(-1)

            act = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(
                int(act) if self.ac.is_discrete else act
            )
            done = terminated or truncated
            ep_ret += reward

            obs_list.append(obs)
            act_list.append(act)
            next_obs_list.append(next_obs)
            done_list.append(done)
            val_list.append(value.item())
            logp_list.append(logp.item())

            # Augment reward with intrinsic signal
            if icm is not None:
                obs_t2 = self._to_tensor(next_obs).unsqueeze(0)
                action_t = action.to(self.device)
                with torch.no_grad():
                    intr, _, _ = icm(obs_t, action_t, obs_t2)
                intr_val = intr.item()
                if self._intr_rms is not None:
                    self._intr_rms.update(intr_val)
                    intr_val = self._intr_rms.normalize(intr_val)
                intr_rewards.append(intr_val)
                reward = reward + self.intrinsic_coeff * intr_val

            rew_list.append(reward)

            if done:
                if episode_returns is not None:
                    episode_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap value for last state
        obs_t = self._to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            last_val = self.ac.get_value(obs_t).item()

        returns, advantages = self._compute_gae(
            rew_list, val_list, done_list, last_val
        )

        return {
            "obs": np.array(obs_list),
            "next_obs": np.array(next_obs_list),
            "actions": np.array(act_list),
            "returns": returns,
            "advantages": advantages,
            "log_probs": np.array(logp_list),
            "intr_reward_mean": float(np.mean(intr_rewards)) if intr_rewards else 0.0,
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, rollout: Dict[str, Any], **_) -> Dict[str, float]:
        obs = self._to_tensor(rollout["obs"])
        next_obs = self._to_tensor(rollout["next_obs"])
        actions = torch.tensor(rollout["actions"], device=self.device)
        returns = torch.tensor(rollout["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout["advantages"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = len(obs)

        total_pg, total_vf, total_ent, total_icm = 0.0, 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = indices[start: start + self.batch_size]
                b_obs = obs[idx]
                b_next_obs = next_obs[idx]
                b_act = actions[idx]
                b_ret = returns[idx]
                b_adv = advantages[idx]
                b_old_logp = old_log_probs[idx]

                dist, value = self.ac(b_obs)
                log_probs = dist.log_prob(b_act)
                if not self.ac.is_discrete:
                    log_probs = log_probs.sum(-1)
                entropy = dist.entropy()
                if not self.ac.is_discrete:
                    entropy = entropy.sum(-1)

                ratio = (log_probs - b_old_logp).exp()
                pg_loss = -torch.min(
                    ratio * b_adv,
                    ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * b_adv,
                ).mean()
                vf_loss = F.mse_loss(value, b_ret)
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coeff * vf_loss + self.ent_coeff * ent_loss

                icm_loss_val = 0.0
                if self.icm is not None:
                    _, icm_loss, _ = self.icm(b_obs, b_act, b_next_obs)
                    loss = loss + icm_loss
                    icm_loss_val = icm_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += ent_loss.item()
                total_icm += icm_loss_val
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            "pg_loss": total_pg / denom,
            "vf_loss": total_vf / denom,
            "ent_loss": total_ent / denom,
            "icm_loss": total_icm / denom,
            "intr_reward_mean": rollout.get("intr_reward_mean", 0.0),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        t = torch.tensor(np.array(x), dtype=torch.float32, device=self.device)
        return t

    def _compute_gae(self, rewards, values, dones, last_value):
        n = len(rewards)
        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_val = last_value
        for t in reversed(range(n)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_val = values[t]
        return returns, advantages

    def save(self, path: str) -> None:
        state = {"ac": self.ac.state_dict()}
        if self.icm is not None:
            state["icm"] = self.icm.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(state["ac"])
        if self.icm is not None and "icm" in state:
            self.icm.load_state_dict(state["icm"])
