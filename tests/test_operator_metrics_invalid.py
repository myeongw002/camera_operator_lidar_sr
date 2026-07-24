import torch

from camera_operator_sr.evaluation.operator_metrics import operator_metric_sums


def test_operator_metrics_exclude_all_invalid_without_nan():
    candidate_valid = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.bool)
    values = operator_metric_sums(torch.tensor([[[[2.0, 0.0]]]]), torch.tensor([[[[3.0, 0.0]]]]), torch.zeros(1, 1, 1, 2), torch.ones(1, 1, 1, 2), torch.tensor([[[[2.0, 4.0], [9.0, 9.0]]]]), candidate_valid, torch.tensor([[[[0.5, 0.5], [0.0, 0.0]]]]), torch.tensor([[[[3.0, 5.0]]]]), torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))
    assert values["operator_query_count"] == 1 and values["all_invalid_positive_query_count"] == 1
    assert all(torch.isfinite(torch.tensor(float(value))) for value in values.values())
