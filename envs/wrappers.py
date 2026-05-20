import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, Wrapper
from functools import partial
from typing import Tuple
import minigrid  # registers MiniGrid envs into gymnasium


class MiniGridImageWrapper(ObservationWrapper):
    """Extract the RGB image from MiniGrid's dict observation and normalize to [0,1].

    MiniGrid returns {'image': ndarray(H,W,3), ...}. This wrapper
    - extracts the 'image' field
    - converts to float32 in [0, 1]
    - transposes to (C, H, W) for PyTorch
    """

    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space["image"].shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

    def observation(self, obs):
        img = obs["image"].astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)


class NormalizeObsWrapper(ObservationWrapper):
    """Normalize flat observations using running mean and std."""

    def __init__(self, env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        obs_dim = int(np.prod(env.observation_space.shape))
        self._mean = np.zeros(obs_dim, dtype=np.float64)
        self._var = np.ones(obs_dim, dtype=np.float64)
        self._count = 0

    def observation(self, obs):
        flat = obs.flatten().astype(np.float64)
        self._count += 1
        delta = flat - self._mean
        self._mean += delta / self._count
        delta2 = flat - self._mean
        self._var += (delta * delta2 - self._var) / self._count
        std = np.sqrt(self._var + self.epsilon)
        return ((flat - self._mean) / std).astype(np.float32).reshape(obs.shape)


class EpisodeStatsWrapper(Wrapper):
    """Track per-episode return and length, exposed via info['episode']."""

    def __init__(self, env):
        super().__init__(env)
        self._ep_return = 0.0
        self._ep_length = 0

    def reset(self, **kwargs):
        self._ep_return = 0.0
        self._ep_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_return += reward
        self._ep_length += 1
        if terminated or truncated:
            info["episode"] = {
                "return": self._ep_return,
                "length": self._ep_length,
            }
        return obs, reward, terminated, truncated, info


def _make_single_env(env_id: str, seed: int, max_steps):
    """Module-level factory (picklable) for AsyncVectorEnv workers."""
    env, _ = make_env(env_id, seed=seed, max_steps=max_steps)
    return env


def make_vec_env(
    env_id: str, n_envs: int = 8, seed: int = 0, max_steps: int = None
) -> Tuple[gym.vector.VectorEnv, str]:
    """Create n_envs parallel environments via AsyncVectorEnv.

    Returns (vec_env, obs_type) where obs_type is 'image' or 'flat'.
    Each worker env i gets seed = seed + i for diverse initial states.
    """
    fns = [partial(_make_single_env, env_id, seed + i, max_steps) for i in range(n_envs)]
    vec_env = gym.vector.AsyncVectorEnv(fns)

    # Detect obs_type from the single observation space
    single_space = vec_env.single_observation_space
    if len(single_space.shape) == 3:
        obs_type = "image"
    else:
        obs_type = "flat"

    return vec_env, obs_type


def make_env(env_id: str, seed: int = 0, render_mode: str = None, max_steps: int = None) -> Tuple[gym.Env, str]:
    """Create and wrap an environment. Returns (env, obs_type).

    obs_type: 'image' | 'flat'
    max_steps: override the environment's default max episode steps.
    """
    make_kwargs = {"render_mode": render_mode}
    if max_steps is not None:
        make_kwargs["max_steps"] = max_steps
    env = gym.make(env_id, **make_kwargs)
    env.reset(seed=seed)

    obs_space = env.observation_space
    if hasattr(obs_space, "spaces") and "image" in obs_space.spaces:
        env = MiniGridImageWrapper(env)
        obs_type = "image"
    elif getattr(obs_space, "shape", None) is not None and len(obs_space.shape) == 1:
        obs_type = "flat"
    else:
        obs_type = "image"

    env = EpisodeStatsWrapper(env)
    return env, obs_type
