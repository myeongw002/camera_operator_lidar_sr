import torch

from camera_operator_sr.models.relation.local_context import robust_normalize_candidates


def test_camera_candidate_normalization_handles_all_valid_counts_and_even_medians():
    values = torch.tensor([[[[1., 3., 5., 7., 9., 11.]]]])
    for count in range(7):
        valid = torch.zeros_like(values, dtype=torch.bool); valid[..., :count] = True
        normalized = robust_normalize_candidates(values, valid)
        assert torch.isfinite(normalized).all()
        assert torch.equal(normalized[~valid], torch.zeros_like(normalized[~valid]))
    valid = torch.tensor([[[[True, True, False, False, False, False]]]])
    normalized = robust_normalize_candidates(values, valid)
    # median(1, 3)=2 and MAD(1, 1)=1
    assert torch.allclose(normalized[..., :2], torch.tensor([[[[-1., 1.]]]]))


def test_camera_candidate_normalization_is_finite_for_equal_values_backward():
    values = torch.ones(1, 1, 1, 6, requires_grad=True)
    normalized = robust_normalize_candidates(values, torch.ones_like(values, dtype=torch.bool))
    normalized.square().sum().backward()
    assert torch.isfinite(values.grad).all()
