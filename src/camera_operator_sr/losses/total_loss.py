from dataclasses import dataclass

from torch import Tensor

from camera_operator_sr.models.outputs import OperatorOutput

from .distillation import bernoulli_kl_loss, operator_kl_loss, residual_kd_loss
from .supervised import residual_regularization, return_loss, supervised_range_loss


@dataclass(frozen=True)
class LossWeights:
    return_weight: float = 1.0
    residual_weight: float = 0.01
    kd_operator: float = 1.0
    kd_return: float = 0.2
    kd_residual: float = 0.05


def supervised_total(output: OperatorOutput, batch: dict, range_mask: Tensor, weights: LossWeights = LossWeights(), return_mask: Tensor | None = None) -> dict[str, Tensor]:
    target = batch["target"]
    return_mask = range_mask if return_mask is None else return_mask
    range_value = supervised_range_loss(output.predicted_range, target["range"], target["valid"], range_mask)
    return_value = return_loss(output.return_logits, target["valid"], return_mask)
    residual_value = residual_regularization(output.residual, output.local_scale, range_mask)
    return {"loss": range_value + weights.return_weight * return_value + weights.residual_weight * residual_value, "range_loss": range_value, "return_loss": return_value, "residual_reg": residual_value}


def distillation_total(student: OperatorOutput, teacher: OperatorOutput, batch: dict, supervised_mask: Tensor, kd_masks, weights: LossWeights = LossWeights()) -> dict[str, Tensor]:
    values = supervised_total(student, batch, supervised_mask, weights)
    operator = operator_kl_loss(student.anchor_logits, teacher.anchor_logits, student.candidate_valid, kd_masks.operator)
    ret = bernoulli_kl_loss(student.return_logits, teacher.return_logits, kd_masks.return_mask)
    residual = residual_kd_loss(student.residual, teacher.residual, kd_masks.residual)
    values.update(kd_operator=operator, kd_return=ret, kd_residual=residual)
    values["loss"] = values["loss"] + weights.kd_operator * operator + weights.kd_return * ret + weights.kd_residual * residual
    return values
