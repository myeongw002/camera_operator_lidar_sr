import torch

from camera_operator_sr.models.relation import CameraGuidedRelationModel, RelationLidarModel
from camera_operator_sr.models.relation.lidar_model import corrected_two_anchor_weights


def batch():
    return {"lidar": {"range": torch.tensor([[[[10., 11., 12., 13.], [30., 31., 32., 33.]]]]), "intensity": torch.ones(1, 1, 2, 4) * .5, "valid": torch.ones(1, 1, 2, 4), "elevation": torch.tensor([-1., 1.]), "azimuth": torch.linspace(-.4, .4, 4)}, "target": {"range": torch.ones(1, 1, 3, 4) * 20., "valid": torch.ones(1, 1, 3, 4), "elevation": torch.tensor([-1., 0., 1.])}, "camera": {"relative_depth": torch.ones(1, 1, 32, 32), "depth_valid": torch.ones(1, 1, 32, 32)}, "calibration": {"K": torch.tensor([[[20., 0., 16.], [0., 20., 16.], [0., 0., 1.]]]), "T_cam_lidar": torch.tensor([[[0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]]]), "image_size": torch.tensor([[32, 32]])}}


def test_fresh_guided_model_equals_frozen_l0_and_only_camera_adapter_trains():
    l0 = RelationLidarModel(); model = CameraGuidedRelationModel(l0); value = batch()
    model.train(); output = model(value); baseline = l0(value)
    assert not model.l0.training and all(not parameter.requires_grad for parameter in model.l0.parameters())
    assert torch.equal(output.camera_correction, torch.zeros_like(output.camera_correction))
    assert torch.equal(output.guided_weights, baseline.final_weights)
    assert torch.equal(output.predicted_range, baseline.predicted_range)
    assert output.camera_candidate_depth.shape == (1, 3, 4, 6)
    assert output.camera_candidate_valid.shape == (1, 3, 4, 6)
    before = {key: value.clone() for key, value in model.l0.state_dict().items()}
    output.predicted_range.sum().backward()
    assert all(parameter.grad is None for parameter in model.l0.parameters())
    optimizer = torch.optim.AdamW(model.camera_adapter.parameters()); optimizer.step()
    assert all(torch.equal(before[key], value) for key, value in model.l0.state_dict().items())


def test_camera_guidance_mask_isolates_correction_and_uses_documented_sign():
    model = CameraGuidedRelationModel(RelationLidarModel()); value = batch()
    with torch.no_grad():
        model.camera_adapter.head[-1].bias.fill_(1.0)
    output = model(value)
    invalid = ~output.camera_guidance_valid.bool()
    assert torch.equal(output.camera_correction[invalid], torch.zeros_like(output.camera_correction[invalid]))
    assert torch.equal(output.predicted_range[invalid], output.lidar.predicted_range[invalid])
    prior = torch.tensor([[[[.5, .5]]]])
    anchor_valid = torch.ones_like(prior, dtype=torch.bool)
    positive = corrected_two_anchor_weights(prior, anchor_valid, torch.ones(1, 1, 1, 1))
    negative = corrected_two_anchor_weights(prior, anchor_valid, -torch.ones(1, 1, 1, 1))
    assert positive[..., 1].item() > positive[..., 0].item()
    assert negative[..., 0].item() > negative[..., 1].item()
