import torch
from torch import Tensor

from camera_operator_sr.types import RangeImage


def uniform_subsample_rows(target: RangeImage, num_input_rows: int = 16) -> tuple[RangeImage, Tensor]:
    h = target.range.shape[-2]
    if not 0 < num_input_rows <= h:
        raise ValueError("num_input_rows must be in [1, target rows]")
    indices = torch.linspace(0, h - 1, num_input_rows, device=target.range.device).round().long()
    return (
        RangeImage(target.range[..., indices, :], target.intensity[..., indices, :], target.valid[..., indices, :]),
        indices,
    )


def sensor_like_subsample_rows(target: RangeImage, input_elevation: Tensor, target_elevation: Tensor) -> tuple[RangeImage, Tensor, Tensor]:
    indices = (input_elevation[:, None] - target_elevation[None, :]).abs().argmin(dim=1)
    if indices.unique().numel() != indices.numel():
        raise ValueError("Sensor-like mapping produced duplicate target rows.")
    exact = torch.isclose(input_elevation, target_elevation[indices], rtol=0.0, atol=1e-5)
    return RangeImage(target.range[..., indices, :], target.intensity[..., indices, :], target.valid[..., indices, :]), indices, exact
