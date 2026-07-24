import math
import torch

from camera_operator_sr.evaluation.evaluator import RangeAccumulator, ReturnAccumulator


def test_accumulators_are_global_count_weighted():
    accumulator = RangeAccumulator()
    accumulator.update(torch.ones(1, 1, 1, 100), torch.zeros(1, 1, 1, 100), torch.ones(1, 1, 1, 100))
    accumulator.update(torch.tensor([[[[10.0]]]]), torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1))
    assert math.isclose(accumulator.result()["range_mae"], 110 / 101)
    returns = ReturnAccumulator(); returns.update(torch.tensor([[[[1.0, 0.0]]]]), torch.tensor([[[[1.0, 1.0]]]]), torch.ones(1, 1, 1, 2))
    assert returns.result()["return_f1"] == 2 / 3 and isinstance(returns.result()["query_count"], int)
