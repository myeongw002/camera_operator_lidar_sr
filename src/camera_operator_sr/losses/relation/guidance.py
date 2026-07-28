"""Guided-only supervision; frozen L0 correction is never regularized here."""
import torch
import torch.nn.functional as F

from camera_operator_sr.training.modules import generated_mask_for


def guided_relation_supervised_loss(output, batch: dict, *, huber_delta: float = .1,
                                    camera_correction_reg_weight: float = 1e-3, eps: float = 1e-6) -> dict:
    target = batch["target"]
    supported = generated_mask_for(batch).bool() & target["valid"].bool() & output.lidar.has_anchor.bool() & output.camera_guidance_valid.bool()
    zero = output.predicted_range.sum() * 0.0
    range_loss = F.smooth_l1_loss(torch.log(output.predicted_range[supported].clamp_min(eps)), torch.log(target["range"][supported].clamp_min(eps)), beta=huber_delta, reduction="mean") if supported.any() else zero
    correction_reg = output.camera_correction[supported].abs().mean() if supported.any() else zero
    return {"loss": range_loss + camera_correction_reg_weight * correction_reg, "total_loss": range_loss + camera_correction_reg_weight * correction_reg, "range_loss": range_loss, "camera_correction_reg": correction_reg, "supported_count": supported.sum()}
