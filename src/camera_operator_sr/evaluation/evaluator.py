"""Count-weighted evaluation accumulators shared by every CSV output."""

from dataclasses import dataclass, field
import math

from torch import Tensor


@dataclass
class RangeAccumulator:
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        selected = mask.bool()
        error = (prediction - target)[selected]
        self.absolute_error_sum += float(error.abs().sum())
        self.squared_error_sum += float(error.square().sum())
        self.count += int(selected.sum())

    def result(self) -> dict:
        return {"range_mae": self.absolute_error_sum / self.count if self.count else 0.0, "range_rmse": math.sqrt(self.squared_error_sum / self.count) if self.count else 0.0, "range_count": self.count}


@dataclass
class ReturnAccumulator:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, probability: Tensor, target_valid: Tensor, mask: Tensor, threshold: float = 0.5) -> None:
        selected, predicted, target = mask.bool(), probability.gt(threshold), target_valid.bool()
        self.tp += int((selected & predicted & target).sum())
        self.fp += int((selected & predicted & ~target).sum())
        self.fn += int((selected & ~predicted & target).sum())
        self.tn += int((selected & ~predicted & ~target).sum())

    def result(self) -> dict:
        precision_denominator, recall_denominator = self.tp + self.fp, self.tp + self.fn
        f1_denominator = 2 * self.tp + self.fp + self.fn
        return {"return_precision": self.tp / precision_denominator if precision_denominator else 0.0, "return_recall": self.tp / recall_denominator if recall_denominator else 0.0, "return_f1": 2 * self.tp / f1_denominator if f1_denominator else 0.0, "hallucination_ratio": self.fp / precision_denominator if precision_denominator else 0.0, "missing_ratio": self.fn / recall_denominator if recall_denominator else 0.0, "return_positive_count": recall_denominator, "return_negative_count": self.tn + self.fp, "query_count": self.tp + self.fp + self.fn + self.tn}


@dataclass
class OperatorAccumulator:
    anchor_abs_error_sum: float = 0.0
    final_abs_error_sum: float = 0.0
    support_error_sum: float = 0.0
    entropy_sum: float = 0.0
    normalized_entropy_sum: float = 0.0
    residual_abs_sum: float = 0.0
    residual_scale_ratio_sum: float = 0.0
    count: int = 0
    all_invalid_positive_count: int = 0
    positive_evaluation_count: int = 0
    all_invalid_all_count: int = 0
    evaluation_count: int = 0

    def update(self, values: dict) -> None:
        self.anchor_abs_error_sum += values["anchor_abs_error_sum"]
        self.final_abs_error_sum += values["final_abs_error_sum"]
        self.support_error_sum += values["support_error_sum"]
        self.entropy_sum += values["entropy_sum"]
        self.normalized_entropy_sum += values["normalized_entropy_sum"]
        self.residual_abs_sum += values["residual_abs_sum"]
        self.residual_scale_ratio_sum += values["residual_scale_ratio_sum"]
        self.count += values["operator_query_count"]
        self.all_invalid_positive_count += values["all_invalid_positive_query_count"]
        self.positive_evaluation_count += values["positive_evaluation_query_count"]
        self.all_invalid_all_count += values["all_invalid_all_query_count"]
        self.evaluation_count += values["evaluation_query_count"]

    def result(self) -> dict:
        count = self.count
        return {"anchor_mae": self.anchor_abs_error_sum / count if count else 0.0, "final_mae": self.final_abs_error_sum / count if count else 0.0, "support_error": self.support_error_sum / count if count else 0.0, "operator_entropy": self.entropy_sum / count if count else 0.0, "operator_normalized_entropy": self.normalized_entropy_sum / count if count else 0.0, "mean_abs_residual": self.residual_abs_sum / count if count else 0.0, "mean_residual_scale_ratio": self.residual_scale_ratio_sum / count if count else 0.0, "operator_query_count": count, "all_invalid_positive_query_count": self.all_invalid_positive_count, "all_invalid_positive_query_ratio": self.all_invalid_positive_count / self.positive_evaluation_count if self.positive_evaluation_count else 0.0, "all_invalid_all_query_count": self.all_invalid_all_count, "all_invalid_all_query_ratio": self.all_invalid_all_count / self.evaluation_count if self.evaluation_count else 0.0}


@dataclass
class MetricGroupAccumulator:
    """Global sums for one beam, distance bin, or named region."""

    range: RangeAccumulator = field(default_factory=RangeAccumulator)
    returns: ReturnAccumulator = field(default_factory=ReturnAccumulator)
    operator: OperatorAccumulator = field(default_factory=OperatorAccumulator)

    def update(self, output: object, target_range: Tensor, target_valid: Tensor, query_mask: Tensor, *, include_return: bool = True) -> None:
        from .operator_metrics import operator_metric_sums

        self.range.update(output.predicted_range, target_range, query_mask * target_valid)
        if include_return:
            self.returns.update(output.return_probability, target_valid, query_mask)
        self.operator.update(operator_metric_sums(output.anchor_range, output.predicted_range, output.residual, output.local_scale, output.candidate_ranges, output.candidate_valid, output.anchor_weights, target_range, target_valid, query_mask))

    def csv_result(self, *, include_return: bool = True, empty_by_range: bool = False) -> dict:
        """Return CSV-safe values: missing denominators are blank, never perfect zero."""
        values = self.range.result() | self.operator.result()
        if include_return:
            values |= self.returns.result()
        query_count = self.returns.result()["query_count"] if include_return else self.range.count
        values["query_count"] = query_count
        values["empty_group"] = bool(self.range.count == 0 if empty_by_range else query_count == 0)
        if self.range.count == 0:
            values["range_mae"] = values["range_rmse"] = None
        if include_return and query_count == 0:
            for key in ("return_precision", "return_recall", "return_f1", "hallucination_ratio", "missing_ratio"):
                values[key] = None
        if self.operator.count == 0:
            for key in ("anchor_mae", "final_mae", "support_error", "operator_entropy", "operator_normalized_entropy", "mean_abs_residual", "mean_residual_scale_ratio"):
                values[key] = None
        if self.operator.positive_evaluation_count == 0:
            values["all_invalid_positive_query_ratio"] = None
        if self.operator.evaluation_count == 0:
            values["all_invalid_all_query_ratio"] = None
        return values
