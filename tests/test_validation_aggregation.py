import math
import pytest
import torch

from camera_operator_sr.training.validation import ValidationRangeAccumulator


def test_global_validation_mae_weights_queries_not_batches():
    accumulator = ValidationRangeAccumulator()
    accumulator.update(torch.ones(1, 1, 1, 100), torch.zeros(1, 1, 1, 100), torch.ones(1, 1, 1, 100))
    accumulator.update(torch.tensor([[[[10.0]]]]), torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1))
    assert accumulator.count == 101
    assert math.isclose(accumulator.score(), 110 / 101)
    assert not math.isclose(accumulator.score(), (1 + 10) / 2)


def test_zero_validation_count_is_an_error():
    with pytest.raises(RuntimeError, match="validation_count"):
        ValidationRangeAccumulator().score()
