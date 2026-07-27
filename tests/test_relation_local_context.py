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


def test_local_normalization_covers_zero_through_six_valid_candidates_and_backward():
    ranges = torch.tensor([[[[2., 2., 4., 8., 16., 32.]]]], requires_grad=True)
    cases = {
        0: torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.bool),
        1: torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool),
        2: torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.bool),
        3: torch.tensor([1, 1, 0, 1, 0, 0], dtype=torch.bool),
        4: torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool),
        6: torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),
    }
    outputs = {}
    for count, mask in cases.items():
        valid = mask.reshape(1, 1, 1, -1)
        output = robust_normalized_log_range(ranges, valid)
        outputs[count] = output
        assert torch.isfinite(output).all()
        assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))
    assert torch.equal(outputs[0], torch.zeros_like(outputs[0]))
    assert outputs[1][0, 0, 0, 0] == 0
    assert torch.allclose(outputs[2][0, 0, 0, [0, 3]], torch.tensor([-1., 1.]), atol=1e-5)
    equal_pair = robust_normalized_log_range(torch.tensor([[[[2., 2.]]]]), torch.ones(1, 1, 1, 2, dtype=torch.bool))
    assert torch.equal(equal_pair, torch.zeros_like(equal_pair))
    # 4 and 6 valid values use averages of their two central sorted values.
    assert torch.allclose(outputs[4][0, 0, 0, :4], torch.tensor([-1., -1., 1., 3.]), atol=1e-5)
    assert torch.allclose(outputs[6][0, 0, 0], torch.tensor([-1., -1., -.333333, .333333, 1., 1.666667]), atol=1e-4)
    sum(value.sum() for value in outputs.values()).backward()
    assert torch.isfinite(ranges.grad).all()
