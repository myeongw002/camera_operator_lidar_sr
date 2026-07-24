import math

import torch
from torch import Tensor


def build_region_masks(azimuth: Tensor, camera_frustum: Tensor | None = None, gt_camera_visible: Tensor | None = None, camera_boundary: Tensor | None = None, transition_width: float = math.radians(5), front_half_width: float = math.radians(45)) -> dict[str, Tensor]:
    """Partition 360° into front-transition, side, and rear; visible is separate."""
    if azimuth.ndim == 2:
        azimuth = azimuth[0]
    angle = torch.atan2(torch.sin(azimuth), torch.cos(azimuth)).abs()
    front = angle <= front_half_width
    transition = (angle > front_half_width) & (angle <= front_half_width + transition_width)
    rear = angle >= math.pi - front_half_width
    side = ~(front | transition | rear)
    width = azimuth.numel()
    regions = {"transition": transition[None, None, None].expand(1, 1, 1, width), "side": side[None, None, None].expand(1, 1, 1, width), "rear": rear[None, None, None].expand(1, 1, 1, width), "full": torch.ones(1, 1, 1, width, dtype=torch.bool, device=azimuth.device)}
    regions["camera_frustum"] = camera_frustum.bool() if camera_frustum is not None else front[None, None, None].expand(1, 1, 1, width)
    regions["gt_camera_visible"] = gt_camera_visible.bool() if gt_camera_visible is not None else torch.zeros_like(regions["camera_frustum"])
    boundary = camera_boundary.bool() if camera_boundary is not None else torch.zeros_like(regions["gt_camera_visible"])
    regions["camera_boundary"] = regions["gt_camera_visible"] & boundary
    regions["camera_interior"] = regions["gt_camera_visible"] & ~boundary
    regions["front_azimuth"] = front[None, None, None].expand_as(regions["camera_frustum"])
    return regions
