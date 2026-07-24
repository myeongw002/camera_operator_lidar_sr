import torch
import torch.nn.functional as F
from torch import Tensor

from .reduction import masked_mean


def operator_kl_loss(student_logits: Tensor, teacher_logits: Tensor, candidate_valid: Tensor, kd_mask: Tensor, temperature: float = 2.0) -> Tensor:
    """Stable KL on valid candidates only.

    Computation is promoted to FP32 so the finite masked sentinel remains safe
    under autocast.  Invalid entries are never multiplied by a log probability.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = candidate_valid.bool()
    any_valid = valid.any(dim=-1)
    sentinel = -1.0e9
    teacher_scaled = (teacher_logits.detach().float() / temperature).masked_fill(~valid, sentinel)
    student_scaled = (student_logits.float() / temperature).masked_fill(~valid, sentinel)
    teacher_probability = torch.softmax(teacher_scaled, dim=-1).masked_fill(~valid, 0.0).detach()
    teacher_log_probability = torch.log_softmax(teacher_scaled, dim=-1)
    student_log_probability = torch.log_softmax(student_scaled, dim=-1)
    terms = teacher_probability * (teacher_log_probability - student_log_probability)
    terms = torch.where(valid, terms, torch.zeros_like(terms))
    per_query = terms.sum(dim=-1) * (temperature**2)
    reduction_mask = kd_mask.squeeze(1).to(per_query.dtype) * any_valid.to(per_query.dtype)
    return (per_query * reduction_mask).sum() / reduction_mask.sum().clamp_min(1e-6)


def bernoulli_kl_loss(student_logits: Tensor, teacher_logits: Tensor, kd_mask: Tensor) -> Tensor:
    teacher_probability = torch.sigmoid(teacher_logits.detach()).clamp(1e-5, 1 - 1e-5)
    student_log_probability = F.logsigmoid(student_logits)
    student_log_not_probability = F.logsigmoid(-student_logits)
    values = teacher_probability * (torch.log(teacher_probability) - student_log_probability) + (1 - teacher_probability) * (torch.log1p(-teacher_probability) - student_log_not_probability)
    return masked_mean(values, kd_mask)


def residual_kd_loss(student_residual: Tensor, teacher_residual: Tensor, kd_mask: Tensor, beta: float = 1.0) -> Tensor:
    return masked_mean(F.huber_loss(student_residual, teacher_residual.detach(), reduction="none", delta=beta), kd_mask)
