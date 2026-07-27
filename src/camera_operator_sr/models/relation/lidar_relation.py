"""Bounded scalar LiDAR interpolation correction."""

from torch import Tensor, nn


class LidarRelationMLP(nn.Module):
    def __init__(self, relation_dim: int, hidden_dim: int = 64, correction_limit: float = 3.0):
        super().__init__()
        if correction_limit <= 0: raise ValueError("correction_limit must be positive")
        self.correction_limit = float(correction_limit)
        self.head = nn.Sequential(nn.Linear(relation_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, relation_feature: Tensor) -> Tensor:
        correction = self.correction_limit * self.head(relation_feature).tanh()
        return correction.permute(0, 3, 1, 2)
