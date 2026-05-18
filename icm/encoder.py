import torch
import torch.nn as nn
from typing import Tuple


class CNNEncoder(nn.Module):
    """CNN encoder for image-based observations (e.g. MiniGrid pixel obs).

    Input:  (B, C, H, W) float tensor in [0, 1]
    Output: (B, feature_dim) feature vector
    """

    def __init__(self, obs_shape: Tuple[int, int, int], feature_dim: int):
        super().__init__()
        c, h, w = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.net(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.feature_dim = feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(obs))


class MLPEncoder(nn.Module):
    """MLP encoder for flat observations (e.g. continuous control).

    Input:  (B, obs_dim) float tensor
    Output: (B, feature_dim) feature vector
    """

    def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.feature_dim = feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def make_encoder(obs_shape, feature_dim: int, encoder_type: str = "auto") -> nn.Module:
    """Factory for encoders. Infers type from obs_shape if encoder_type='auto'."""
    if encoder_type == "auto":
        encoder_type = "cnn" if len(obs_shape) == 3 else "mlp"

    if encoder_type == "cnn":
        assert len(obs_shape) == 3, "CNN encoder requires (C, H, W) obs_shape"
        return CNNEncoder(obs_shape, feature_dim)
    elif encoder_type == "mlp":
        obs_dim = obs_shape[0] if len(obs_shape) == 1 else int(torch.tensor(obs_shape).prod())
        return MLPEncoder(obs_dim, feature_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
