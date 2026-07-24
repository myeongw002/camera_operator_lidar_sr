import pytest
import torch

from camera_operator_sr.data.subsampling import sensor_like_subsample_rows
from camera_operator_sr.types import RangeImage


def test_sensor_like_subsampling_rejects_duplicate_mapping():
    target = RangeImage(torch.ones(1, 2, 4), torch.ones(1, 2, 4), torch.ones(1, 2, 4))
    with pytest.raises(ValueError, match="duplicate"):
        sensor_like_subsample_rows(target, torch.tensor([0.0, 0.01]), torch.tensor([0.0, 1.0]))
