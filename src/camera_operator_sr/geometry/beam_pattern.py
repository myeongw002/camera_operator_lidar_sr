import torch
from torch import Tensor


def find_vertical_bracketing_beams(input_elevation: Tensor, target_elevation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return lower/upper input row indices and whether a target lies in input FOV.

    The input table may be ascending or descending, but must not contain duplicate angles.
    """
    if input_elevation.ndim != 1 or target_elevation.ndim != 1:
        raise ValueError("elevation tensors must be one-dimensional")
    order = torch.argsort(input_elevation)
    sorted_angles = input_elevation[order]
    upper_position = torch.searchsorted(sorted_angles, target_elevation)
    valid = (upper_position > 0) & (upper_position < sorted_angles.numel())
    lower_position = (upper_position - 1).clamp(0, sorted_angles.numel() - 1)
    upper_position = upper_position.clamp(0, sorted_angles.numel() - 1)
    exact = torch.isclose(target_elevation[:, None], input_elevation[None, :], rtol=0.0, atol=1e-5).any(dim=1)
    nearest = (target_elevation[:, None] - input_elevation[None, :]).abs().argmin(dim=1)
    lower, upper = order[lower_position], order[upper_position]
    lower = torch.where(exact, nearest, lower)
    upper = torch.where(exact, nearest, upper)
    return lower, upper, valid | exact
