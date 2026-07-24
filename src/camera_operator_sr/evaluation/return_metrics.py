import torch
from torch import Tensor


def return_metrics(probability: Tensor, target_valid: Tensor, mask: Tensor, threshold: float = 0.5) -> dict[str, Tensor]:
    selected = mask.bool()
    predicted = probability.gt(threshold) & selected
    actual = target_valid.bool() & selected
    tp = (predicted & actual).sum()
    fp = (predicted & ~actual).sum()
    fn = (~predicted & actual).sum()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    return {"precision": precision, "recall": recall, "hallucinated_return_ratio": fp / (tp + fp).clamp_min(1), "missing_return_ratio": fn / (tp + fn).clamp_min(1)}
