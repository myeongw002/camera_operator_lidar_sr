import torch
from torch import Tensor


def range_metrics(prediction: Tensor, target: Tensor, mask: Tensor) -> dict[str, Tensor]:
    valid = mask.bool()
    error = prediction - target
    count = valid.sum().clamp_min(1)
    return {"mae": error.abs()[valid].sum() / count, "rmse": error.square()[valid].sum().div(count).sqrt(), "count": count}
