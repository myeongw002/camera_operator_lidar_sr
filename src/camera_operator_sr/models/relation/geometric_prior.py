"""Elevation-based two-anchor geometric interpolation."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PriorOutput:
    """Geometric-prior tensors with anchor order ``[lower, upper]``."""

    weights: Tensor             # [B,H,W,2]
    anchor_ranges: Tensor       # [B,H,W,2]
    anchor_valid: Tensor        # [B,H,W,2]
    has_anchor: Tensor          # [B,1,H,W]
    predicted_range: Tensor     # [B,1,H,W]


class VerticalLinearPrior(nn.Module):
    """Use actual elevation fraction to interpolate lower/upper LiDAR anchors."""

    def forward(self, anchor_ranges: Tensor, anchor_valid: Tensor, query_fraction: Tensor) -> PriorOutput:
        if anchor_ranges.ndim != 4 or anchor_ranges.shape[-1] != 2:
            raise ValueError("anchor_ranges must be [B,H,W,2] in [lower, upper] order")
        if anchor_valid.shape != anchor_ranges.shape:
            raise ValueError("anchor_valid must match anchor_ranges")
        batch, height, width, _ = anchor_ranges.shape
        fraction = self._expand_fraction(query_fraction, batch, height, width, anchor_ranges)
        geometric = torch.stack((1.0 - fraction, fraction), dim=-1)
        valid = anchor_valid.bool()
        lower_valid, upper_valid = valid.unbind(dim=-1)
        both_valid = lower_valid & upper_valid
        lower_only = lower_valid & ~upper_valid
        upper_only = ~lower_valid & upper_valid
        weights = torch.zeros_like(geometric)
        weights = torch.where(both_valid[..., None], geometric, weights)
        weights[..., 0] = torch.where(lower_only, torch.ones_like(weights[..., 0]), weights[..., 0])
        weights[..., 1] = torch.where(upper_only, torch.ones_like(weights[..., 1]), weights[..., 1])
        has_anchor = valid.any(dim=-1)
        prediction = (weights * anchor_ranges).sum(dim=-1)
        prediction = torch.where(has_anchor, prediction, torch.zeros_like(prediction))
        return PriorOutput(weights, anchor_ranges, valid, has_anchor[:, None], prediction[:, None])

    @staticmethod
    def _expand_fraction(query_fraction: Tensor, batch: int, height: int, width: int, reference: Tensor) -> Tensor:
        fraction = query_fraction.to(device=reference.device, dtype=reference.dtype)
        if fraction.shape == (height,):
            return fraction[None, :, None].expand(batch, -1, width)
        if fraction.shape == (batch, height):
            return fraction[:, :, None].expand(-1, -1, width)
        if fraction.shape == (batch, height, width):
            return fraction
        if fraction.shape == (batch, height, width, 1):
            return fraction.squeeze(-1)
        raise ValueError("query_fraction must be [H], [B,H], [B,H,W], or [B,H,W,1]")
