import torch
from torch import Tensor


def operator_metric_sums(anchor_range: Tensor, predicted_range: Tensor, residual: Tensor, local_scale: Tensor, candidate_ranges: Tensor, candidate_valid: Tensor, weights: Tensor, target_range: Tensor, target_valid: Tensor, evaluation_mask: Tensor) -> dict[str, float | int]:
    """Additive finite statistics; all-invalid rows never enter operator means."""
    evaluation = evaluation_mask.squeeze(1).bool()
    positive = evaluation & target_valid.squeeze(1).bool()
    valid = candidate_valid.bool()
    has_candidate = valid.any(dim=-1)
    selected = positive & has_candidate
    target = target_range.squeeze(1)
    min_range = candidate_ranges.masked_fill(~valid, float("inf")).min(dim=-1).values
    max_range = candidate_ranges.masked_fill(~valid, float("-inf")).max(dim=-1).values
    nearest = (candidate_ranges - target[..., None]).abs().masked_fill(~valid, float("inf")).min(dim=-1).values
    support = torch.where((target >= min_range) & (target <= max_range), torch.zeros_like(nearest), nearest)
    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
    valid_count = valid.sum(dim=-1)
    normalized_entropy = torch.where(valid_count > 1, entropy / torch.log(valid_count.to(entropy.dtype)), torch.zeros_like(entropy))

    def summed(value: Tensor) -> float:
        # Mask before reduction, avoiding the 0 * inf all-invalid failure mode.
        return float(torch.nan_to_num(torch.where(selected, value, torch.zeros_like(value)), nan=0.0, posinf=0.0, neginf=0.0).sum())

    residual_scale_ratio = residual.squeeze(1).abs() / local_scale.squeeze(1).clamp_min(1e-8)
    return {"anchor_abs_error_sum": summed((anchor_range.squeeze(1) - target).abs()), "final_abs_error_sum": summed((predicted_range.squeeze(1) - target).abs()), "support_error_sum": summed(support), "entropy_sum": summed(entropy), "normalized_entropy_sum": summed(normalized_entropy), "residual_abs_sum": summed(residual.squeeze(1).abs()), "residual_scale_ratio_sum": summed(residual_scale_ratio), "operator_query_count": int(selected.sum()), "all_invalid_positive_query_count": int((positive & ~has_candidate).sum()), "positive_evaluation_query_count": int(positive.sum()), "all_invalid_all_query_count": int((evaluation & ~has_candidate).sum()), "evaluation_query_count": int(evaluation.sum())}
