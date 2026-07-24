import torch
import torch.nn.functional as F
from torch import Tensor


def build_range_boundary_mask(ranges: Tensor, valid: Tensor, abs_threshold: float = 0.5, relative_threshold: float = 0.02, dilation_pixels: int = 1) -> Tensor:
    """Valid-pair range discontinuities, without invalid-edge artefacts."""
    horizontal = torch.zeros_like(ranges); vertical = torch.zeros_like(ranges)
    next_range, next_valid = torch.roll(ranges, -1, dims=-1), torch.roll(valid, -1, dims=-1)
    hpair = valid.bool() & next_valid.bool()
    vpair = valid[..., :-1, :].bool() & valid[..., 1:, :].bool()
    horizontal = (next_range - ranges).abs() * hpair
    vertical[..., :-1, :] = (ranges[..., 1:, :] - ranges[..., :-1, :]).abs() * vpair
    threshold = abs_threshold + relative_threshold * ranges
    boundary = torch.maximum(horizontal, vertical).gt(threshold) & valid.bool()
    if dilation_pixels:
        boundary = F.max_pool2d(boundary.float(), 2 * dilation_pixels + 1, stride=1, padding=dilation_pixels).bool() & valid.bool()
    return boundary
