import math
from types import SimpleNamespace

import torch

from camera_operator_sr.evaluation.evaluator import MetricGroupAccumulator


def test_operator_csv_statistics_handle_all_candidate_cases():
    # Four positive queries: all valid, partial, one valid, and all-invalid;
    # plus one negative all-invalid query to make the two failure ratios differ.
    target = torch.tensor([[[[2.0, 4.0, 6.0, 8.0, 0.0]]]])
    valid_target = torch.tensor([[[[1, 1, 1, 1, 0]]]], dtype=torch.bool)
    candidate_ranges = torch.tensor([[[[2.0, 3.0], [4.0, 9.0], [6.0, 9.0], [0.0, 0.0], [0.0, 0.0]]]])
    candidate_valid = torch.tensor([[[[1, 1], [1, 0], [1, 0], [0, 0], [0, 0]]]], dtype=torch.bool)
    weights = torch.tensor([[[[0.5, 0.5], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]])
    output = SimpleNamespace(
        anchor_range=torch.tensor([[[[2.5, 4.0, 6.0, 0.0, 0.0]]]]),
        predicted_range=torch.tensor([[[[2.5, 4.0, 6.0, 0.0, 0.0]]]]),
        residual=torch.zeros(1, 1, 1, 5), local_scale=torch.ones(1, 1, 1, 5),
        candidate_ranges=candidate_ranges, candidate_valid=candidate_valid, anchor_weights=weights,
        return_probability=torch.zeros(1, 1, 1, 5),
    )
    accumulator = MetricGroupAccumulator()
    accumulator.update(output, target, valid_target, torch.ones_like(target, dtype=torch.bool))
    values = accumulator.csv_result()
    assert values["operator_query_count"] == 3
    assert values["all_invalid_positive_query_count"] == 1
    assert values["all_invalid_all_query_count"] == 2
    assert math.isclose(values["all_invalid_positive_query_ratio"], 1 / 4)
    assert math.isclose(values["all_invalid_all_query_ratio"], 2 / 5)
    for key in ("anchor_mae", "final_mae", "support_error", "operator_entropy", "operator_normalized_entropy"):
        assert math.isfinite(values[key])
    assert isinstance(values["operator_query_count"], int)


def test_operator_empty_group_is_not_serialized_as_perfect_score():
    values = MetricGroupAccumulator().csv_result()
    assert values["empty_group"] is True
    assert values["anchor_mae"] is None
