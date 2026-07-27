import torch

from camera_operator_sr.models.relation import GeometricBaselineModel, VerticalLinearPrior
from camera_operator_sr.training.modules import generated_mask_for


def test_vertical_prior_uses_fraction_and_handles_every_anchor_validity_case():
    prior = VerticalLinearPrior()
    ranges = torch.tensor([[[[10.0, 30.0], [10.0, 30.0], [10.0, 30.0], [10.0, 30.0]]]])
    valid = torch.tensor([[[[1, 1], [1, 0], [0, 1], [0, 0]]]], dtype=torch.bool)
    output = prior(ranges, valid, torch.tensor([0.25]))
    assert output.weights.tolist() == [[[[0.75, 0.25], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]]
    assert output.predicted_range.tolist() == [[[[15.0, 10.0, 30.0, 0.0]]]]
    assert output.has_anchor.tolist() == [[[[True, True, True, False]]]]


def test_geometric_baseline_output_shapes_zero_correction_and_observed_row_preservation():
    batch = {
        "lidar": {
            "range": torch.tensor([[[[10.0, 11.0, 12.0, 13.0], [30.0, 31.0, 32.0, 33.0]]]]),
            "valid": torch.ones(1, 1, 2, 4, dtype=torch.bool),
            "elevation": torch.tensor([-1.0, 1.0]),
            "azimuth": torch.linspace(-torch.pi, torch.pi, 4),
        },
        "target": {
            "elevation": torch.tensor([-1.0, 0.0, 1.0]),
            "range": torch.zeros(1, 1, 3, 4),
        },
    }
    output = GeometricBaselineModel()(batch)
    assert output.predicted_range.shape == (1, 1, 3, 4)
    assert output.prior_weights.shape == output.final_weights.shape == (1, 3, 4, 2)
    assert output.anchor_ranges.shape == output.anchor_valid.shape == (1, 3, 4, 2)
    assert output.has_anchor.shape == output.correction.shape == (1, 1, 3, 4)
    assert torch.equal(output.prior_weights, output.final_weights)
    assert torch.count_nonzero(output.correction) == 0
    assert torch.equal(output.predicted_range[:, :, 0], batch["lidar"]["range"][:, :, 0])
    assert torch.equal(output.predicted_range[:, :, 2], batch["lidar"]["range"][:, :, 1])
    assert torch.equal(generated_mask_for(batch)[0, 0, :, 0].bool(), torch.tensor([False, True, False]))


def test_geometric_baseline_all_invalid_anchors_are_zero():
    batch = {
        "lidar": {
            "range": torch.ones(1, 1, 2, 2), "valid": torch.zeros(1, 1, 2, 2, dtype=torch.bool),
            "elevation": torch.tensor([-1.0, 1.0]), "azimuth": torch.tensor([-1.0, 1.0]),
        },
        "target": {"elevation": torch.tensor([0.0])},
    }
    output = GeometricBaselineModel()(batch)
    assert not output.has_anchor.any()
    assert torch.count_nonzero(output.predicted_range) == 0
