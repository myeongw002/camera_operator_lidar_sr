import torch

from camera_operator_sr.models.relation import GeometricBaselineModel, RelationLidarModel, corrected_two_anchor_weights


def batch(valid=True):
    return {
        "lidar": {"range": torch.tensor([[[[10., 11., 12., 13.], [30., 31., 32., 33.]]]]), "intensity": torch.ones(1, 1, 2, 4) * .5, "valid": torch.full((1, 1, 2, 4), valid), "elevation": torch.tensor([-1., 1.]), "azimuth": torch.linspace(-1., 1., 4)},
        "target": {"range": torch.ones(1, 1, 3, 4) * 20., "valid": torch.ones(1, 1, 3, 4), "elevation": torch.tensor([-1., 0., 1.])},
    }


def test_l0_output_and_b0_support_match_and_zero_correction_equivalence():
    value = batch(); l0, b0 = RelationLidarModel(), GeometricBaselineModel()
    output, prior = l0(value), b0(value)
    assert output.predicted_range.shape == (1, 1, 3, 4)
    assert output.relation_feature.shape[-1] == l0.aggregator.relation_dim
    assert torch.equal(output.has_anchor, prior.has_anchor)
    assert torch.count_nonzero(output.correction) == 0
    assert torch.allclose(output.predicted_range, prior.predicted_range)
    assert torch.isfinite(output.predicted_range).all()


def test_correction_sign_and_invalid_anchor_rules():
    prior = torch.tensor([[[[.5, .5], [1., 0.], [0., 0.]]]])
    valid = torch.tensor([[[[1, 1], [1, 0], [0, 0]]]], dtype=torch.bool)
    positive = corrected_two_anchor_weights(prior, valid, torch.tensor([[[[1., 1., 1.]]]]))
    negative = corrected_two_anchor_weights(prior, valid, torch.tensor([[[[-1., -1., -1.]]]]))
    assert positive[0, 0, 0, 1] > .5 and negative[0, 0, 0, 0] > .5
    assert torch.equal(positive[..., 1, :], prior[..., 1, :]) and torch.equal(positive[..., 2, :], prior[..., 2, :])


def test_l0_backward_is_finite_for_all_invalid_input():
    value = batch(valid=False); output = RelationLidarModel()(value); output.predicted_range.sum().backward()
    assert torch.isfinite(output.predicted_range).all() and not output.has_anchor.any()
