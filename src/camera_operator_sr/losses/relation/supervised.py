"""Supported-query log-range supervision for L0."""

import torch
import torch.nn.functional as F

from camera_operator_sr.training.modules import generated_mask_for


def relation_supervised_loss(output, batch: dict, *, huber_delta: float = 0.1, correction_reg_weight: float = 1e-3, eps: float = 1e-6) -> dict:
    """Return finite L0 loss; correction regularization includes only two-anchor queries."""
    target = batch["target"]
    generated = generated_mask_for(batch).bool()
    supported = generated & target["valid"].bool() & output.has_anchor.bool()
    both_anchor = supported.squeeze(1) & output.anchor_valid.bool().all(dim=-1)
    zero = output.predicted_range.sum() * 0.0
    if supported.any():
        range_loss = F.smooth_l1_loss(torch.log(output.predicted_range[supported].clamp_min(eps)), torch.log(target["range"][supported].clamp_min(eps)), beta=huber_delta, reduction="mean")
    else:
        range_loss = zero
    if both_anchor.any():
        correction_reg = output.correction.squeeze(1)[both_anchor].abs().mean()
    else:
        correction_reg = zero
    total = range_loss + correction_reg_weight * correction_reg
    return {
        "loss": total, "total_loss": total, "range_loss": range_loss,
        "correction_reg": correction_reg, "supported_count": supported.sum(),
        "both_anchor_count": both_anchor.sum(),
    }
