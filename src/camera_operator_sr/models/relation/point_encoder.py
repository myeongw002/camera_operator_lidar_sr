"""Shared candidate-wise point MLP."""

import torch
from torch import Tensor, nn


class RelationPointEncoder(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        if tokens.ndim != 5 or valid.shape != tokens.shape[:-1]:
            raise ValueError("tokens must be [B,H,W,K,F] and valid [B,H,W,K]")
        return self.net(tokens) * valid[..., None].to(tokens.dtype)
