import torch
from torch import Tensor

from .projection import project_lidar_points


def build_gt_visible_valid_mask(target_xyz: Tensor, target_valid: Tensor, K: Tensor, T_cam_lidar: Tensor, image_size: tuple[int, int], abs_tolerance: float = 0.05, relative_tolerance: float = 0.01) -> Tensor:
    """GT-only z-buffer visibility mask for training/evaluation, never model input."""
    b, h, w, _ = target_xyz.shape
    image_h, image_w = image_size
    projection = project_lidar_points(target_xyz.reshape(b, h * w, 3), K, T_cam_lidar, image_size)
    valid = projection.valid & target_valid.reshape(b, h * w).bool()
    pixels = projection.uv[..., 1].floor().long().clamp(0, image_h - 1) * image_w + projection.uv[..., 0].floor().long().clamp(0, image_w - 1)
    zmin = torch.full((b, image_h * image_w), float("inf"), device=target_xyz.device, dtype=target_xyz.dtype)
    for batch_index in range(b):
        zmin[batch_index].scatter_reduce_(0, pixels[batch_index, valid[batch_index]], projection.camera_depth[batch_index, valid[batch_index]], reduce="amin", include_self=True)
    closest = zmin.gather(1, pixels)
    tolerance = abs_tolerance + relative_tolerance * projection.camera_depth
    visible = valid & (projection.camera_depth - closest).abs().le(tolerance)
    return visible.reshape(b, 1, h, w).to(target_xyz.dtype)


def build_gt_camera_visibility(*args, **kwargs) -> Tensor:
    """Compatibility alias for the old API; use build_gt_visible_valid_mask."""
    return build_gt_visible_valid_mask(*args, **kwargs)


def build_camera_query_frustum_mask(target_elevation: Tensor, azimuth: Tensor, K: Tensor, T_cam_lidar: Tensor, image_size: tuple[int, int], range_samples: Tensor | None = None) -> Tensor:
    """Return [B,1,H,W] where a target ray intersects the camera frustum.

    This does not depend on target returns, so it is safe to use for negative
    return supervision and return distillation.
    """
    if range_samples is None:
        range_samples = torch.tensor([2.0, 5.0, 10.0, 20.0, 40.0, 80.0], device=K.device, dtype=K.dtype)
    if target_elevation.ndim == 2:
        target_elevation = target_elevation[0]
    if azimuth.ndim == 2:
        azimuth = azimuth[0]
    elevation_grid, azimuth_grid = torch.meshgrid(target_elevation.to(K.device), azimuth.to(K.device), indexing="ij")
    directions = torch.stack((torch.cos(elevation_grid) * torch.cos(azimuth_grid), torch.cos(elevation_grid) * torch.sin(azimuth_grid), torch.sin(elevation_grid)), dim=-1)
    b, h, w = K.shape[0], directions.shape[0], directions.shape[1]
    points = (directions[None, :, :, None, :] * range_samples.to(K.device, K.dtype)[None, None, None, :, None]).expand(b, -1, -1, -1, -1)
    projection = project_lidar_points(points.reshape(b, h * w * range_samples.numel(), 3), K, T_cam_lidar, image_size)
    return projection.valid.reshape(b, h, w, range_samples.numel()).any(dim=-1).unsqueeze(1).to(K.dtype)
