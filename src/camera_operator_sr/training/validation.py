from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ValidationRangeAccumulator:
    """Global query-weighted range MAE accumulator used for checkpoint selection."""
    absolute_error_sum: float = 0.0
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        selected = mask.bool()
        self.absolute_error_sum += float(((prediction - target).abs() * selected.to(prediction.dtype)).sum())
        self.count += int(selected.sum())

    def score(self) -> float:
        if self.count == 0:
            raise RuntimeError("validation_count is zero; refusing to select a best checkpoint")
        return float(self.absolute_error_sum / self.count)
