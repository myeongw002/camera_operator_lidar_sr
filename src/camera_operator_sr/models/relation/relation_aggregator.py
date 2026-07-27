"""Safe local relation pooling for the L0 correction head."""

import torch
from torch import Tensor, nn


class RelationAggregator(nn.Module):
    def __init__(self, embedding_dim: int, lower_center_slot: int = 1, upper_center_slot: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lower_center_slot, self.upper_center_slot = lower_center_slot, upper_center_slot
        self.relation_dim = 6 * embedding_dim + 6

    def forward(self, embeddings: Tensor, candidate_valid: Tensor, anchor_valid: Tensor, prior_weights: Tensor, query_fraction: Tensor, anchor_normalized_log_range: Tensor) -> Tensor:
        if embeddings.ndim != 5 or candidate_valid.shape != embeddings.shape[:-1]:
            raise ValueError("embedding and candidate mask shapes are incompatible")
        valid = candidate_valid.bool()
        count = valid.sum(dim=-1, keepdim=True)
        mean = (embeddings * valid[..., None].to(embeddings.dtype)).sum(dim=-2) / count.clamp_min(1).to(embeddings.dtype)
        maximum = embeddings.masked_fill(~valid[..., None], float("-inf")).max(dim=-2).values
        maximum = torch.where(count.squeeze(-1)[..., None].gt(0), maximum, torch.zeros_like(maximum))
        lower = embeddings[..., self.lower_center_slot, :] * anchor_valid[..., 0, None].to(embeddings.dtype)
        upper = embeddings[..., self.upper_center_slot, :] * anchor_valid[..., 1, None].to(embeddings.dtype)
        difference = upper - lower
        entropy = -(prior_weights * torch.log(prior_weights.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        fraction = _expand_fraction(query_fraction, embeddings.shape[:3], embeddings)
        normalized_difference = (anchor_normalized_log_range[..., 1] - anchor_normalized_log_range[..., 0]).unsqueeze(-1)
        valid_ratio = count.to(embeddings.dtype) / valid.shape[-1]
        pattern = anchor_valid.to(embeddings.dtype)
        return torch.cat((mean, maximum, lower, upper, difference, difference.abs(), fraction, normalized_difference, valid_ratio, entropy, pattern), dim=-1)


def _expand_fraction(fraction: Tensor, shape: tuple[int, int, int], reference: Tensor) -> Tensor:
    batch, height, width = shape
    value = fraction.to(reference.device, reference.dtype)
    if value.shape == (height,): return value[None, :, None, None].expand(batch, -1, width, -1)
    if value.shape == (batch, height): return value[:, :, None, None].expand(-1, -1, width, -1)
    if value.shape == (batch, height, width): return value[..., None]
    raise ValueError("query_fraction must be [H], [B,H], or [B,H,W]")
