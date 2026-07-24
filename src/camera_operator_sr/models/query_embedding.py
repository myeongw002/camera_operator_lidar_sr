import torch
import torch.nn as nn
from torch import Tensor


class RelativeGeometryEmbedding(nn.Module):
    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))

    def forward(self, delta_elevation: Tensor, delta_azimuth: Tensor, candidate_range: Tensor, candidate_valid: Tensor) -> Tensor:
        log_range = torch.log(candidate_range.clamp_min(1e-3))
        x = torch.stack((delta_elevation, torch.sin(delta_azimuth), torch.cos(delta_azimuth), log_range, candidate_valid.to(log_range.dtype)), dim=-1)
        return self.net(x)
