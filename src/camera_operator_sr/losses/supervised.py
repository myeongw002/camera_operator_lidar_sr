import torch
import torch.nn.functional as F
from torch import Tensor

from .reduction import masked_mean


def supervised_range_loss(prediction: Tensor, target: Tensor, target_valid: Tensor, generated_mask: Tensor, beta: float = 1.0) -> Tensor:
    mask = target_valid.bool() & generated_mask.bool()
    return masked_mean(F.huber_loss(prediction, target, reduction="none", delta=beta), mask)


def return_loss(logits: Tensor, target_valid: Tensor, generated_mask: Tensor, pos_weight: float | None = None) -> Tensor:
    weight = None if pos_weight is None else torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    values = F.binary_cross_entropy_with_logits(logits, target_valid.to(logits.dtype), reduction="none", pos_weight=weight)
    return masked_mean(values, generated_mask)


def residual_regularization(residual: Tensor, local_scale: Tensor, mask: Tensor) -> Tensor:
    return masked_mean(residual.abs() / local_scale.clamp_min(1e-6), mask)
