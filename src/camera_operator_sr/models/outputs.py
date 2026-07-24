from dataclasses import dataclass

from torch import Tensor


@dataclass
class OperatorOutput:
    anchor_logits: Tensor
    anchor_weights: Tensor
    candidate_ranges: Tensor
    candidate_valid: Tensor
    has_candidate: Tensor      # [B, 1, H_target, W]
    anchor_range: Tensor
    residual: Tensor
    predicted_range: Tensor
    return_logits: Tensor
    return_probability: Tensor
    local_scale: Tensor
