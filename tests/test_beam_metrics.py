import csv
import math
from types import SimpleNamespace

import torch

from camera_operator_sr.evaluation.beam_metrics import beam_row_metadata, target_row_mask
from camera_operator_sr.evaluation.evaluator import MetricGroupAccumulator


def _output(prediction: torch.Tensor) -> SimpleNamespace:
    batch, _, rows, width = prediction.shape
    candidate_ranges = prediction.squeeze(1)[..., None]
    valid = torch.ones_like(candidate_ranges, dtype=torch.bool)
    return SimpleNamespace(
        predicted_range=prediction,
        return_probability=torch.ones_like(prediction),
        anchor_range=prediction,
        residual=torch.zeros_like(prediction),
        local_scale=torch.ones_like(prediction),
        candidate_ranges=candidate_ranges,
        candidate_valid=valid,
        anchor_weights=torch.ones_like(candidate_ranges),
    )


def test_beam_global_aggregation_and_empty_serialization():
    # The one target row has 100 errors of 1 and one error of 10.  A frame
    # average would be 5.5; the global numerator/denominator is 110 / 101.
    accumulator = MetricGroupAccumulator()
    first = torch.ones(1, 1, 1, 100)
    second = torch.full((1, 1, 1, 1), 10.0)
    accumulator.update(_output(first), torch.zeros_like(first), torch.ones_like(first), target_row_mask(1, 1, 100, 0, device=first.device))
    accumulator.update(_output(second), torch.zeros_like(second), torch.ones_like(second), target_row_mask(1, 1, 1, 0, device=second.device))
    values = accumulator.csv_result()
    assert math.isclose(values["range_mae"], 110 / 101)
    assert not math.isclose(values["range_mae"], 5.5)
    assert values["range_count"] == 101 and isinstance(values["range_count"], int)
    assert values["query_count"] == 101 and isinstance(values["query_count"], int)

    empty = MetricGroupAccumulator().csv_result()
    assert empty["empty_group"] is True
    assert empty["range_mae"] is None and empty["operator_entropy"] is None


def test_beam_flags_use_elevation_mapping_not_row_identity():
    metadata = beam_row_metadata(torch.tensor([-0.1, 0.2]), torch.tensor([-0.1, 0.0, 0.2]))
    assert [entry["is_observed_row"] for entry in metadata] == [True, False, True]
    assert [entry["is_generated_row"] for entry in metadata] == [False, True, False]
    assert metadata[1]["target_elevation_deg"] == 0.0
