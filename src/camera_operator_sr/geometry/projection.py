from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ProjectionResult:
    uv: Tensor
    camera_depth: Tensor
    valid: Tensor
    normalized_grid: Tensor


def project_lidar_points(points_lidar: Tensor, K: Tensor, T_cam_lidar: Tensor, image_size: tuple[int, int], align_corners: bool = False) -> ProjectionResult:
    """Project [B,N,3] points; grid convention is explicitly matched to grid_sample."""
    if points_lidar.ndim != 3 or points_lidar.shape[-1] != 3:
        raise ValueError("points_lidar must be [B, N, 3]")
    height, width = image_size
    b, n, _ = points_lidar.shape
    homogeneous = torch.cat((points_lidar, torch.ones((b, n, 1), device=points_lidar.device, dtype=points_lidar.dtype)), dim=-1)
    camera = homogeneous @ T_cam_lidar.transpose(-1, -2)
    xyz = camera[..., :3]
    depth = xyz[..., 2]
    projected = xyz @ K.transpose(-1, -2)
    uv = projected[..., :2] / depth.clamp_min(torch.finfo(points_lidar.dtype).eps)[..., None]
    valid = depth.gt(0) & uv[..., 0].ge(0) & uv[..., 0].lt(width) & uv[..., 1].ge(0) & uv[..., 1].lt(height)
    if align_corners:
        x = 2 * uv[..., 0] / max(width - 1, 1) - 1
        y = 2 * uv[..., 1] / max(height - 1, 1) - 1
    else:
        x = 2 * (uv[..., 0] + 0.5) / width - 1
        y = 2 * (uv[..., 1] + 0.5) / height - 1
    return ProjectionResult(uv, depth, valid, torch.stack((x, y), dim=-1))
