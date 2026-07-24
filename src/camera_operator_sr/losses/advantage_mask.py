from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DistillationMasks:
    operator: Tensor
    residual: Tensor
    return_mask: Tensor

@dataclass(frozen=True)
class AdvantageConfig:
    mode: str = "soft"
    range_margin: float = 0.1
    range_temperature: float = 0.1
    return_margin: float = 0.05
    return_temperature: float = 0.1


def _advantage(baseline_error: Tensor, teacher_error: Tensor, margin: float, temperature: float, mode: str) -> Tensor:
    if mode == "none": return torch.ones_like(baseline_error)
    if mode == "hard": return (teacher_error + margin < baseline_error).to(baseline_error.dtype)
    if mode != "soft": raise ValueError(f"unknown advantage mode: {mode}")
    return torch.sigmoid((baseline_error - teacher_error - margin) / max(temperature, 1e-6))


def compute_range_advantage(baseline_range: Tensor, teacher_range: Tensor, target_range: Tensor, target_valid: Tensor, margin: float = 0.1, temperature: float = 0.1, mode: str = "soft") -> Tensor:
    baseline_error = (baseline_range - target_range).abs()
    teacher_error = (teacher_range.detach() - target_range).abs()
    advantage = _advantage(baseline_error, teacher_error, margin, temperature, mode)
    return advantage * target_valid.to(advantage.dtype)


def compute_return_advantage(baseline_logits: Tensor, teacher_logits: Tensor, target_valid: Tensor, margin: float = 0.05, temperature: float = 0.1, mode: str = "soft") -> Tensor:
    target = target_valid.to(baseline_logits.dtype)
    baseline_error = F.binary_cross_entropy_with_logits(baseline_logits, target, reduction="none")
    teacher_error = F.binary_cross_entropy_with_logits(teacher_logits.detach(), target, reduction="none")
    return _advantage(baseline_error, teacher_error, margin, temperature, mode)


def build_distillation_masks(generated_mask: Tensor, gt_visible_valid_mask: Tensor, camera_query_frustum_mask: Tensor, target_valid: Tensor, range_advantage: Tensor, return_advantage: Tensor, has_candidate: Tensor | None = None) -> DistillationMasks:
    eligible = 1.0 if has_candidate is None else has_candidate.to(range_advantage.dtype)
    operator = generated_mask.to(range_advantage.dtype) * gt_visible_valid_mask.to(range_advantage.dtype) * target_valid.to(range_advantage.dtype) * eligible * range_advantage
    return_mask = generated_mask.to(return_advantage.dtype) * camera_query_frustum_mask.to(return_advantage.dtype) * eligible * return_advantage
    return DistillationMasks(operator=operator, residual=operator, return_mask=return_mask)
