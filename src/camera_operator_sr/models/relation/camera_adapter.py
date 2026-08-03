"""Small masked camera relation adapter; no attention, CNN, or absolute azimuth.

Only camera-valid candidates enter masked pooling.  Consequently the token's
camera-valid and LiDAR-valid fields are normally one for pooled candidates;
they remain in the fixed nine-value checkpoint contract for diagnostics and a
future PR4 ablation of camera-valid-only versus LiDAR-valid pooling.
"""
import torch
from torch import Tensor, nn


class CameraRelationAdapter(nn.Module):
    def __init__(self, l0_relation_dim: int, token_dim: int = 9, point_hidden_dim: int = 16,
                 relation_hidden_dim: int = 32, correction_limit: float = 3.0):
        super().__init__()
        self.correction_limit = float(correction_limit)
        self.point = nn.Sequential(nn.Linear(token_dim, point_hidden_dim), nn.SiLU(), nn.Linear(point_hidden_dim, point_hidden_dim), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(l0_relation_dim + 2 * point_hidden_dim, relation_hidden_dim), nn.SiLU(), nn.Linear(relation_hidden_dim, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, tokens: Tensor, valid: Tensor, l0_feature: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.point(tokens) * valid[..., None].to(tokens.dtype)
        count = valid.sum(-1, keepdim=True)
        mean = embedded.sum(-2) / count.clamp_min(1).to(tokens.dtype)
        maximum = embedded.masked_fill(~valid[..., None], float("-inf")).max(-2).values
        maximum = torch.where(count.gt(0), maximum, torch.zeros_like(maximum))
        feature = torch.cat((l0_feature, mean, maximum), dim=-1)
        return (self.correction_limit * self.head(feature).tanh()).permute(0, 3, 1, 2), feature
