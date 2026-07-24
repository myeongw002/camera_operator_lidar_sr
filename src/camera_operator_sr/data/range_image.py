import math

import torch
from torch import Tensor

from camera_operator_sr.types import RangeImage


def pointcloud_to_range_image(
    points_xyz: Tensor,
    intensity: Tensor | None,
    elevation_angles: Tensor,
    width: int,
    collision_policy: str = "nearest",
    row_tolerance: float | None = None,
) -> RangeImage:
    """Rasterise one LiDAR scan using azimuth bin centres over [-pi, pi)."""
    if collision_policy != "nearest":
        raise ValueError("only nearest collision policy is implemented")
    if points_xyz.ndim != 2 or points_xyz.shape[-1] != 3:
        raise ValueError("points_xyz must be [N, 3]")
    if width <= 0 or elevation_angles.ndim != 1:
        raise ValueError("invalid range-image geometry")
    h = elevation_angles.numel()
    device, dtype = points_xyz.device, points_xyz.dtype
    ranges = torch.linalg.vector_norm(points_xyz, dim=-1)
    horizontal = torch.linalg.vector_norm(points_xyz[:, :2], dim=-1)
    elevation = torch.atan2(points_xyz[:, 2], horizontal)
    azimuth = torch.atan2(points_xyz[:, 1], points_xyz[:, 0])
    rows = (elevation[:, None] - elevation_angles[None, :]).abs().argmin(dim=1)
    cols = torch.floor((azimuth + math.pi) * width / (2 * math.pi)).long().remainder(width)
    keep = torch.isfinite(ranges) & ranges.gt(0)
    if row_tolerance is not None:
        keep &= (elevation - elevation_angles[rows]).abs().le(row_tolerance)
    flat = rows * width + cols
    out = torch.full((h * width,), float("inf"), dtype=dtype, device=device)
    if keep.any():
        out.scatter_reduce_(0, flat[keep], ranges[keep], reduce="amin", include_self=True)
    valid = torch.isfinite(out)
    range_image = torch.where(valid, out, torch.zeros_like(out)).view(1, h, width)
    intensity_image = torch.zeros((h * width,), dtype=dtype, device=device)
    if intensity is not None and keep.any():
        # Assign intensity of a nearest point deterministically after range reduction.
        nearest = keep & torch.isclose(ranges, out[flat], rtol=0.0, atol=1e-6)
        intensity_image.scatter_reduce_(0, flat[nearest], intensity[nearest].to(dtype), reduce="amax", include_self=True)
    return RangeImage(range_image, intensity_image.view(1, h, width), valid.view(1, h, width).to(dtype))


def range_image_to_pointcloud(ranges: Tensor, valid: Tensor, elevation: Tensor, azimuth: Tensor) -> Tensor:
    """Convert [..., H, W] range values to [..., H, W, 3], zeroing invalid pixels."""
    if ranges.shape[-2:] != (elevation.numel(), azimuth.numel()):
        raise ValueError("range image and angle dimensions disagree")
    phi = elevation.reshape(*([1] * (ranges.ndim - 2)), -1, 1)
    theta = azimuth.reshape(*([1] * (ranges.ndim - 2)), 1, -1)
    x = ranges * torch.cos(phi) * torch.cos(theta)
    y = ranges * torch.cos(phi) * torch.sin(theta)
    z = ranges * torch.sin(phi)
    return torch.stack((x, y, z), dim=-1) * valid.to(ranges.dtype)[..., None]
