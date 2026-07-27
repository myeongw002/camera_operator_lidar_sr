from dataclasses import dataclass

import torch
from torch import Tensor

from .beam_pattern import find_vertical_bracketing_beams


def expected_candidate_count(horizontal_radius: int) -> int:
    """Candidate slots emitted by ``build_candidate_index`` for every query."""
    if horizontal_radius < 0:
        raise ValueError("horizontal_radius must be non-negative")
    return 2 * (2 * horizontal_radius + 1)


@dataclass(frozen=True)
class CandidateIndex:
    row_indices: Tensor       # [H_target, K]
    column_offsets: Tensor    # [K]
    geometric_valid: Tensor   # [H_target, K]
    delta_elevation: Tensor   # [H_target, K]
    delta_azimuth: Tensor     # [K]
    lower_center_slot: int
    upper_center_slot: int
    query_fraction: Tensor    # [H_target], 0 at lower beam and 1 at upper beam

    @property
    def candidate_count(self) -> int:
        return self.column_offsets.numel()


def build_candidate_index(input_elevation: Tensor, target_elevation: Tensor, width: int, horizontal_radius: int = 1) -> CandidateIndex:
    if width <= 0 or horizontal_radius < 0:
        raise ValueError("width must be positive and radius non-negative")
    lower, upper, in_fov = find_vertical_bracketing_beams(input_elevation, target_elevation)
    offsets = torch.arange(-horizontal_radius, horizontal_radius + 1, device=input_elevation.device)
    rows = torch.stack((lower[:, None].expand(-1, offsets.numel()), upper[:, None].expand(-1, offsets.numel())), dim=1).flatten(1)
    column_offsets = offsets.repeat(2)
    geometric_valid = in_fov[:, None].expand_as(rows).clone()
    # Exact observed rows need one vertical source, not duplicate lower/upper candidates.
    duplicate = lower.eq(upper)[:, None].expand(-1, offsets.numel())
    geometric_valid[:, offsets.numel():] &= ~duplicate
    delta_elevation = input_elevation[rows] - target_elevation[:, None]
    delta_azimuth = column_offsets.to(dtype=input_elevation.dtype) * (2 * torch.pi / width)
    lower_elevation, upper_elevation = input_elevation[lower], input_elevation[upper]
    elevation_span = upper_elevation - lower_elevation
    # ``lower`` and ``upper`` are ordered by physical elevation, even when the
    # input range image stores rows in descending elevation order.  Exact
    # observed rows use one source (lower == upper), hence their fraction is 0.
    query_fraction = torch.where(
        elevation_span.abs().gt(torch.finfo(input_elevation.dtype).eps),
        (target_elevation - lower_elevation).div(elevation_span).clamp(0.0, 1.0),
        torch.zeros_like(target_elevation),
    )
    lower_center_slot = horizontal_radius
    upper_center_slot = 3 * horizontal_radius + 1
    return CandidateIndex(
        rows.long(), column_offsets.long(), geometric_valid, delta_elevation,
        delta_azimuth, lower_center_slot, upper_center_slot, query_fraction,
    )


def gather_candidate_values(values: Tensor, candidate_index: CandidateIndex) -> Tensor:
    """Gather [B,C,H_in,W] values to [B,H_target,W,K,C] with horizontal wrapping."""
    if values.ndim != 4:
        raise ValueError("values must be [B, C, H_in, W]")
    b, c, h, w = values.shape
    rows = candidate_index.row_indices
    if rows.numel() and (rows.min() < 0 or rows.max() >= h):
        raise ValueError("candidate rows are incompatible with values")
    target_h, k = rows.shape
    base_cols = torch.arange(w, device=values.device)[None, :, None]
    cols = (base_cols + candidate_index.column_offsets.to(values.device)[None, None, :]).remainder(w)
    selected_rows = rows.to(values.device)[:, None, :].expand(target_h, w, k)
    flat_indices = (selected_rows * w + cols).reshape(-1)
    flattened = values.permute(0, 2, 3, 1).reshape(b, h * w, c)
    return flattened[:, flat_indices].reshape(b, target_h, w, k, c)
