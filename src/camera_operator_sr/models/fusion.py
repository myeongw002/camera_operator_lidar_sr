import torch
import torch.nn as nn
from torch import Tensor


class ObservedAnchorFusion(nn.Module):
    def __init__(self, lidar_feature_dim: int = 96, depth_feature_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(lidar_feature_dim + depth_feature_dim + 1, lidar_feature_dim, 1),
            nn.GroupNorm(min(8, lidar_feature_dim), lidar_feature_dim),
            nn.SiLU(),
            nn.Conv2d(lidar_feature_dim, lidar_feature_dim, 1),
        )

    def forward(self, lidar_feature: Tensor, depth_feature: Tensor, confidence: Tensor) -> Tensor:
        return self.net(torch.cat((lidar_feature, depth_feature * confidence, confidence), dim=1))
