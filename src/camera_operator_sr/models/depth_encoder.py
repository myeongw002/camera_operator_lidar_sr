import torch.nn as nn
from torch import Tensor


class DepthEncoder(nn.Module):
    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, feature_dim, 3, stride=2, padding=1), nn.GroupNorm(min(8, feature_dim), feature_dim), nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.GroupNorm(min(8, feature_dim), feature_dim), nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
