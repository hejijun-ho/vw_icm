from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch.nn as nn


class BaseAlgorithm(ABC):
    """Abstract base for RL algorithms that support an optional ICM module."""

    @abstractmethod
    def collect_rollout(self, env, icm: Optional[nn.Module] = None) -> Dict[str, Any]:
        """Collect a rollout from the environment, augmenting rewards with ICM if provided."""

    @abstractmethod
    def update(self, rollout: Dict[str, Any]) -> Dict[str, float]:
        """Perform a policy/value update from collected rollout data. Returns logging info."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from path."""
