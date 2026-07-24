import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common.circular_conv import CircularConv2d


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            CircularConv2d(in_channels, out_channels, 3, stride=stride),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            CircularConv2d(out_channels, out_channels, 3),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class LidarEncoder(nn.Module):
    """Horizontal U-Net; vertical resolution is preserved throughout."""
    def __init__(self, in_channels: int = 5, feature_dim: int = 96):
        super().__init__()
        self.stem = ConvBlock(in_channels, 32)
        self.down1 = ConvBlock(32, 64, (1, 2))
        self.down2 = ConvBlock(64, feature_dim, (1, 2))
        self.up1 = ConvBlock(feature_dim + 64, feature_dim)
        self.up2 = ConvBlock(feature_dim + 32, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        width = x.shape[-1]
        stem = self.stem(x)
        down1 = self.down1(stem)
        down2 = self.down2(down1)
        up1 = F.interpolate(down2, size=down1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(torch.cat((up1, down1), dim=1))
        up2 = F.interpolate(up1, size=(stem.shape[-2], width), mode="bilinear", align_corners=False)
        return self.up2(torch.cat((up2, stem), dim=1))
