"""Observed-candidate XYZ construction for camera guidance.

Slots preserve CandidateIndex order: lower[-1,0,+1], upper[-1,0,+1].
"""
import torch
from torch import Tensor

from .candidate_graph import CandidateIndex


def candidate_lidar_points(ranges: Tensor, valid: Tensor, index: CandidateIndex,
                           input_elevation: Tensor, azimuth: Tensor) -> tuple[Tensor, Tensor]:
    """Return observed candidate XYZ [B,H,W,K,3] and valid mask [B,H,W,K]."""
    batch, height, width, slots = ranges.shape
    elevation = input_elevation[0] if input_elevation.ndim == 2 else input_elevation
    candidate_elevation = elevation[index.row_indices.to(elevation.device)][None, :, None, :].expand(batch, -1, width, -1)
    base_azimuth = azimuth if azimuth.ndim == 2 else azimuth[None].expand(batch, -1)
    if base_azimuth.shape != (batch, width):
        raise ValueError("azimuth must be [W] or [B,W] matching candidate ranges")
    # Gather the actual source columns, including the circular seam.  This is
    # intentionally not reconstructed from a nominal angular increment.
    columns = (torch.arange(width, device=ranges.device)[:, None]
               + index.column_offsets.to(ranges.device)[None, :]).remainder(width)
    candidate_azimuth = base_azimuth[:, columns].reshape(batch, width, slots)
    candidate_azimuth = candidate_azimuth[:, None].expand(-1, height, -1, -1)
    horizontal = ranges * torch.cos(candidate_elevation)
    xyz = torch.stack((horizontal * torch.cos(candidate_azimuth), horizontal * torch.sin(candidate_azimuth), ranges * torch.sin(candidate_elevation)), dim=-1)
    mask = valid.bool() & index.geometric_valid.to(valid.device)[None, :, None, :]
    return xyz * mask[..., None].to(xyz.dtype), mask
