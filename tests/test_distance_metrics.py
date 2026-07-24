import math

import pytest
import torch

from camera_operator_sr.evaluation.distance_metrics import gt_distance_bin_mask, parse_distance_bins, serialize_distance_boundary


def test_distance_bins_use_gt_half_open_intervals_and_inf():
    target_range = torch.tensor([[[[5.0, 10.0, 19.9, 20.0, 39.9, 40.0, 80.0, 100.0]]]])
    target_valid = torch.tensor([[[[1, 1, 1, 1, 1, 1, 1, 0]]]], dtype=torch.bool)
    bins = parse_distance_bins(["0", "10", "20", "40", "60", "80", "inf"])
    counts = [int(gt_distance_bin_mask(target_range, target_valid, low, high).sum()) for low, high in zip(bins, bins[1:])]
    assert counts == [1, 2, 2, 1, 0, 1]
    # Binning does not inspect a predicted range tensor: valid 100 m would be
    # in the final bin if it were valid, regardless of any prediction.
    assert serialize_distance_boundary(bins[-1]) == "inf"


@pytest.mark.parametrize("values", [["0"], ["0", "10", "10"], ["-1", "10"], ["0", "inf", "20"]])
def test_distance_bin_validation(values):
    with pytest.raises(ValueError):
        parse_distance_bins(values)
