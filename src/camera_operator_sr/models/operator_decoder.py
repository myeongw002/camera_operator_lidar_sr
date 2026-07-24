import torch
import torch.nn as nn
from torch import Tensor

from camera_operator_sr.geometry.candidate_graph import CandidateIndex, gather_candidate_values

from .common.masked_ops import masked_softmax
from .outputs import OperatorOutput
from .query_embedding import RelativeGeometryEmbedding


class OperatorDecoder(nn.Module):
    def __init__(self, feature_dim: int = 96, geometry_dim: int = 32, hidden_dim: int = 128, residual_scale_alpha: float = 1.5, residual_min: float = 0.1, residual_max: float = 3.0):
        super().__init__()
        self.geometry = RelativeGeometryEmbedding(geometry_dim)
        self.anchor_head = nn.Sequential(nn.Linear(feature_dim + geometry_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.residual_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.return_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.residual_scale_alpha = residual_scale_alpha
        self.residual_min = residual_min
        self.residual_max = residual_max

    def forward(self, anchor_features: Tensor, input_range: Tensor, input_valid: Tensor, candidate_index: CandidateIndex) -> OperatorOutput:
        candidate_features = gather_candidate_values(anchor_features, candidate_index)
        candidate_ranges = gather_candidate_values(input_range, candidate_index).squeeze(-1)
        observed_valid = gather_candidate_values(input_valid, candidate_index).squeeze(-1).bool()
        geometric_valid = candidate_index.geometric_valid.to(input_valid.device)[None, :, None, :]
        candidate_valid = observed_valid & geometric_valid
        h_target, width, k = candidate_ranges.shape[1:]
        delta_elevation = candidate_index.delta_elevation.to(input_range.device)[None, :, None, :].expand(input_range.shape[0], -1, width, -1)
        delta_azimuth = candidate_index.delta_azimuth.to(input_range.device)[None, None, None, :].expand(input_range.shape[0], h_target, width, -1)
        geometry = self.geometry(delta_elevation, delta_azimuth, candidate_ranges, candidate_valid)
        logits = self.anchor_head(torch.cat((candidate_features, geometry), dim=-1)).squeeze(-1)
        weights = masked_softmax(logits, candidate_valid)
        anchor_range = (weights * candidate_ranges).sum(dim=-1, keepdim=True)
        weighted_feature = (weights[..., None] * candidate_features).sum(dim=-2)
        masked_ranges = candidate_ranges.masked_fill(~candidate_valid, float("nan"))
        median = torch.nanmedian(masked_ranges, dim=-1, keepdim=True).values
        mad = torch.nanmedian((masked_ranges - median).abs(), dim=-1, keepdim=True).values
        local_scale = torch.nan_to_num(self.residual_scale_alpha * mad, nan=self.residual_min).clamp(self.residual_min, self.residual_max).detach()
        residual = local_scale * torch.tanh(self.residual_head(weighted_feature))
        has_candidate = candidate_valid.any(dim=-1, keepdim=True)
        anchor_range = torch.where(has_candidate, anchor_range, torch.zeros_like(anchor_range))
        residual = torch.where(has_candidate, residual, torch.zeros_like(residual))
        predicted_range = (anchor_range + residual).clamp_min(0).permute(0, 3, 1, 2)
        anchor_range = anchor_range.permute(0, 3, 1, 2)
        residual = residual.permute(0, 3, 1, 2)
        local_scale = local_scale.permute(0, 3, 1, 2)
        return_logits = self.return_head(weighted_feature).masked_fill(~has_candidate, -20.0).permute(0, 3, 1, 2)
        return OperatorOutput(logits, weights, candidate_ranges, candidate_valid, has_candidate.permute(0, 3, 1, 2), anchor_range, residual, predicted_range, return_logits, torch.sigmoid(return_logits), local_scale)
