"""Helpers for target-row (beam) evaluation."""

import torch
from torch import Tensor

from camera_operator_sr.data.masks import build_generated_row_mask


def beam_row_metadata(input_elevation: Tensor, target_elevation: Tensor) -> list[dict[str, int | float | bool]]:
    """Describe target rows using the same observed/generated rule as training."""
    if input_elevation.ndim != 1 or target_elevation.ndim != 1:
        raise ValueError("Beam metadata requires one shared input and target elevation grid.")
    generated = build_generated_row_mask(input_elevation, target_elevation).reshape(-1).bool()
    return [
        {
            "target_row": row,
            "target_elevation_deg": float(torch.rad2deg(target_elevation[row])),
            "is_observed_row": bool(~generated[row]),
            "is_generated_row": bool(generated[row]),
        }
        for row in range(target_elevation.numel())
    ]


def target_row_mask(batch_size: int, target_rows: int, width: int, row: int, *, device: torch.device) -> Tensor:
    if not 0 <= row < target_rows:
        raise IndexError(f"Target row {row} is outside [0, {target_rows}).")
    mask = torch.zeros((batch_size, 1, target_rows, width), dtype=torch.bool, device=device)
    mask[:, :, row, :] = True
    return mask
