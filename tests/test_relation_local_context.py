import torch

from camera_operator_sr.models.relation.local_context import robust_normalized_log_range


def test_local_normalization_is_finite_for_validity_edge_cases():
    ranges = torch.tensor([[[[2., 2., 8., 0.]]]])
    for valid in (torch.tensor([[[[1, 1, 1, 0]]]], dtype=torch.bool), torch.tensor([[[[0, 1, 0, 0]]]], dtype=torch.bool), torch.zeros(1, 1, 1, 4, dtype=torch.bool)):
        output = robust_normalized_log_range(ranges, valid)
        assert torch.isfinite(output).all() and torch.equal(output[~valid], torch.zeros_like(output[~valid]))


def test_even_count_uses_symmetric_middle_average():
    output = robust_normalized_log_range(torch.tensor([[[[2., 8.]]]]), torch.ones(1, 1, 1, 2, dtype=torch.bool))
    assert torch.allclose(output, torch.tensor([[[[-1., 1.]]]]), atol=1e-5)
