import torch
from torch import Tensor


def build_generated_row_mask(input_elevation: Tensor, target_elevation: Tensor, exact_tolerance: float = 1e-5) -> Tensor:
    """Return [1, H_target, 1]; observed rows are zero, generated rows one."""
    if input_elevation.ndim != 1 or target_elevation.ndim != 1:
        raise ValueError("elevation tensors must be one-dimensional")
    exact = (target_elevation[:, None] - input_elevation[None, :]).abs().le(exact_tolerance).any(dim=1)
    return (~exact).to(dtype=torch.float32)[None, :, None]


def expand_row_mask(row_mask: Tensor, width: int, batch_size: int) -> Tensor:
    return row_mask[None].expand(batch_size, 1, -1, width)
