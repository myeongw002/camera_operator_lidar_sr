"""Metrics for comparing an L0 relation prior with its final correction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


def build_prior_prediction(output) -> Tensor:
    """Return the B0 prediction represented by a relation-model output."""
    prediction = (output.prior_weights * output.anchor_ranges).sum(dim=-1)
    return torch.where(output.has_anchor.squeeze(1).bool(), prediction, torch.zeros_like(prediction))[:, None]


def masked_weight_entropy(weights: Tensor, mask: Tensor) -> tuple[float, int]:
    """Return the entropy sum and selected query count without empty-group bias."""
    selected = mask.squeeze(1).bool()
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
    return float(entropy[selected].sum()), int(selected.sum())


def anchor_selection_statistics(output, target_range: Tensor, mask: Tensor) -> tuple[int, int]:
    """Count cases where L0 prefers the GT-closer anchor (two anchors only)."""
    target = target_range.squeeze(1)
    valid = output.anchor_valid.bool()
    errors = (output.anchor_ranges - target[..., None]).abs()
    eligible = mask.squeeze(1).bool() & valid.all(dim=-1) & ~torch.isclose(errors[..., 0], errors[..., 1])
    selected = output.final_weights.argmax(dim=-1) == errors.argmin(dim=-1)
    return int((selected & eligible).sum()), int(eligible.sum())


@dataclass
class RelationMetricAccumulator:
    """Count-weighted prior/final metrics for one evaluation group."""

    prior_absolute_error_sum: float = 0.0
    prior_squared_error_sum: float = 0.0
    final_absolute_error_sum: float = 0.0
    final_squared_error_sum: float = 0.0
    correction_absolute_sum: float = 0.0
    prior_entropy_sum: float = 0.0
    final_entropy_sum: float = 0.0
    supported_count: int = 0
    valid_target_count: int = 0
    unsupported_valid_target_count: int = 0
    anchor_selection_correct: int = 0
    anchor_selection_count: int = 0

    def update(self, output, target_range: Tensor, target_valid: Tensor, query_mask: Tensor) -> None:
        query = query_mask.bool()
        valid_target = query & target_valid.bool()
        supported = valid_target & output.has_anchor.bool()
        prior = build_prior_prediction(output)
        final_error = (output.predicted_range - target_range)[supported]
        prior_error = (prior - target_range)[supported]
        self.prior_absolute_error_sum += float(prior_error.abs().sum())
        self.prior_squared_error_sum += float(prior_error.square().sum())
        self.final_absolute_error_sum += float(final_error.abs().sum())
        self.final_squared_error_sum += float(final_error.square().sum())
        self.correction_absolute_sum += float(output.correction[supported].abs().sum())
        prior_entropy, count = masked_weight_entropy(output.prior_weights, supported)
        final_entropy, final_count = masked_weight_entropy(output.final_weights, supported)
        if count != final_count:
            raise RuntimeError("relation entropy masks disagree")
        self.prior_entropy_sum += prior_entropy
        self.final_entropy_sum += final_entropy
        self.supported_count += count
        self.valid_target_count += int(valid_target.sum())
        self.unsupported_valid_target_count += int((valid_target & ~output.has_anchor.bool()).sum())
        correct, eligible = anchor_selection_statistics(output, target_range, supported)
        self.anchor_selection_correct += correct
        self.anchor_selection_count += eligible

    def result(self) -> dict:
        count = self.supported_count
        if not count:
            return {
                "prior_range_mae": None, "prior_range_rmse": None,
                "final_range_mae": None, "final_range_rmse": None,
                "mae_improvement": None, "rmse_improvement": None,
                "supported_count": 0, "valid_target_count": self.valid_target_count,
                "unsupported_valid_target_count": self.unsupported_valid_target_count,
                "anchor_coverage": None if not self.valid_target_count else 0.0,
                "mean_abs_correction": None, "prior_weight_entropy": None,
                "final_weight_entropy": None, "anchor_selection_accuracy": None,
                "anchor_selection_count": self.anchor_selection_count, "empty_group": True,
            }
        prior_mae = self.prior_absolute_error_sum / count
        final_mae = self.final_absolute_error_sum / count
        prior_rmse = math.sqrt(self.prior_squared_error_sum / count)
        final_rmse = math.sqrt(self.final_squared_error_sum / count)
        return {
            "prior_range_mae": prior_mae, "prior_range_rmse": prior_rmse,
            "final_range_mae": final_mae, "final_range_rmse": final_rmse,
            "mae_improvement": prior_mae - final_mae,
            "rmse_improvement": prior_rmse - final_rmse,
            "supported_count": count, "valid_target_count": self.valid_target_count,
            "unsupported_valid_target_count": self.unsupported_valid_target_count,
            "anchor_coverage": count / self.valid_target_count if self.valid_target_count else None,
            "mean_abs_correction": self.correction_absolute_sum / count,
            "prior_weight_entropy": self.prior_entropy_sum / count,
            "final_weight_entropy": self.final_entropy_sum / count,
            "anchor_selection_accuracy": self.anchor_selection_correct / self.anchor_selection_count if self.anchor_selection_count else None,
            "anchor_selection_count": self.anchor_selection_count, "empty_group": False,
        }
