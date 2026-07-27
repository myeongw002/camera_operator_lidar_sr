"""Local 2x3 LiDAR candidate features without absolute-position shortcuts."""

from dataclasses import dataclass

import torch
from torch import Tensor

from camera_operator_sr.geometry.candidate_graph import CandidateIndex, gather_candidate_values


@dataclass
class LocalCandidateContext:
    ranges: Tensor                 # [B,H,W,K]
    intensity: Tensor              # [B,H,W,K]
    valid: Tensor                  # [B,H,W,K]
    normalized_log_range: Tensor   # [B,H,W,K]
    tokens: Tensor                 # [B,H,W,K,9]


def robust_normalized_log_range(ranges: Tensor, valid: Tensor, *, eps: float = 1e-6, clamp: float = 5.0) -> Tensor:
    """Median/MAD-normalize valid ranges, averaging the two middle values for even counts."""
    if ranges.shape != valid.shape:
        raise ValueError("ranges and valid must have the same shape")
    log_ranges = torch.log(ranges.clamp_min(eps))
    finite_log = torch.nan_to_num(log_ranges, nan=0.0, posinf=0.0, neginf=0.0)
    count = valid.sum(dim=-1, keepdim=True)
    ordered = finite_log.masked_fill(~valid.bool(), float("inf")).sort(dim=-1).values
    lower_index = ((count - 1).clamp_min(0) // 2).long()
    upper_index = (count.clamp_min(1) // 2).long()
    median = .5 * (ordered.gather(-1, lower_index) + ordered.gather(-1, upper_index)).masked_fill(count.eq(0), 0.0)
    deviation = (finite_log - median).abs()
    ordered_deviation = deviation.masked_fill(~valid.bool(), float("inf")).sort(dim=-1).values
    mad = (.5 * (ordered_deviation.gather(-1, lower_index) + ordered_deviation.gather(-1, upper_index))).masked_fill(count.eq(0), 0.0)
    normalized = (finite_log - median) / (mad + eps)
    return torch.where(valid.bool(), normalized.clamp(-clamp, clamp), torch.zeros_like(normalized))


def build_local_candidate_context(input_range: Tensor, input_intensity: Tensor, input_valid: Tensor, index: CandidateIndex) -> LocalCandidateContext:
    """Gather 2x3 candidates and construct the nine shared-MLP token features."""
    ranges = gather_candidate_values(input_range, index).squeeze(-1)
    intensity = gather_candidate_values(input_intensity, index).squeeze(-1)
    observed_valid = gather_candidate_values(input_valid, index).squeeze(-1).bool()
    geometric_valid = index.geometric_valid.to(input_valid.device)[None, :, None, :]
    valid = observed_valid & geometric_valid
    normalized_log_range = robust_normalized_log_range(ranges, valid)
    batch, height, width, candidates = ranges.shape
    delta_elevation = index.delta_elevation.to(device=ranges.device, dtype=ranges.dtype)[None, :, None, :].expand(batch, -1, width, -1)
    delta_azimuth = index.delta_azimuth.to(device=ranges.device, dtype=ranges.dtype)[None, None, None, :].expand(batch, height, width, -1)
    slots = torch.arange(candidates, device=ranges.device)
    upper = slots.ge(candidates // 2).to(ranges.dtype)[None, None, None, :].expand(batch, height, width, -1)
    center = ((slots.eq(index.lower_center_slot)) | (slots.eq(index.upper_center_slot))).to(ranges.dtype)[None, None, None, :].expand(batch, height, width, -1)
    radius = max(int(index.column_offsets.abs().max().item()), 1)
    horizontal_offset = index.column_offsets.to(device=ranges.device, dtype=ranges.dtype).div(radius)[None, None, None, :].expand(batch, height, width, -1)
    # Prepared KITTI intensity is the original float LiDAR reflectance (normally
    # [0,1]); clamping prevents outliers from overwhelming the initial MLP.
    normalized_intensity = torch.nan_to_num(intensity, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    tokens = torch.stack((normalized_log_range, normalized_intensity, valid.to(ranges.dtype), delta_elevation, torch.sin(delta_azimuth), torch.cos(delta_azimuth), upper, center, horizontal_offset), dim=-1)
    return LocalCandidateContext(ranges, intensity, valid, normalized_log_range, tokens * valid[..., None].to(tokens.dtype))
