import torch
from torch import Tensor


def masked_mean(values: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(eps)
