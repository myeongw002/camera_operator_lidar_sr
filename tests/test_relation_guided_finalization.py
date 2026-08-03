import pytest
import torch

from camera_operator_sr.models.relation import CameraGuidedRelationModel, RelationLidarModel
from camera_operator_sr.training.checkpoint import save_checkpoint
from test_relation_guided_model import batch


def _sample():
    return {"lidar": {"elevation": torch.tensor([-1., 1.]), "azimuth": torch.linspace(-1., 1., 4)}, "target": {"elevation": torch.tensor([-1., 0., 1.])}}


def test_camera_sampling_zeros_invalid_projection_depth_valid_and_nonfinite_depth():
    model, value = CameraGuidedRelationModel(RelationLidarModel()), batch()
    value["calibration"]["K"] = torch.tensor([[[5., 0., 16.], [0., 5., 16.], [0., 0., 1.]]])
    state = model.l0.build_state(value)
    sampled, valid, _ = model._camera_samples(value, state)
    assert valid.any() and torch.isfinite(sampled).all()
    for mutate in (
        lambda current: current["camera"]["depth_valid"].zero_(),
        lambda current: current["camera"]["relative_depth"].fill_(float("nan")),
        lambda current: current["calibration"]["K"].__setitem__((..., 0, 2), 1e6),
        lambda current: current["calibration"]["T_cam_lidar"].__setitem__((..., 2, 0), -1.0),
    ):
        current = {key: {name: tensor.clone() for name, tensor in item.items()} if isinstance(item, dict) else item for key, item in value.items()}
        mutate(current)
        depth, mask, _ = model._camera_samples(current, model.l0.build_state(current))
        assert not mask.any()
        assert torch.equal(depth, torch.zeros_like(depth))


@pytest.mark.parametrize("invalid_slots", [(1,), (4,), (1, 4), (1, 4)], ids=("lower_center", "upper_center", "both_centers", "side_only"))
def test_guidance_requires_both_center_camera_slots(monkeypatch, invalid_slots):
    model, value = CameraGuidedRelationModel(RelationLidarModel()), batch()

    def samples(_batch, state):
        shape = state["context"].valid.shape
        valid = torch.ones(shape, dtype=torch.bool)
        for slot in invalid_slots:
            valid[..., slot] = False
        return torch.zeros(shape), valid, torch.ones(shape, dtype=torch.bool)

    monkeypatch.setattr(model, "_camera_samples", samples)
    output = model(value)
    assert not output.camera_guidance_valid.any()
    assert torch.equal(output.camera_correction, torch.zeros_like(output.camera_correction))
    assert torch.equal(output.predicted_range, output.lidar.predicted_range)


def test_one_and_zero_anchor_queries_leave_guided_output_equal_to_l0():
    model, value = CameraGuidedRelationModel(RelationLidarModel()), batch()
    for row_count in (1, 2):
        current = {key: {name: tensor.clone() for name, tensor in item.items()} if isinstance(item, dict) else item for key, item in value.items()}
        current["lidar"]["valid"][:, :, :row_count] = 0
        output = model(current)
        assert torch.equal(output.camera_correction, torch.zeros_like(output.camera_correction))
        assert torch.equal(output.guided_weights, output.lidar.final_weights)
        assert torch.equal(output.predicted_range, output.lidar.predicted_range)


def test_guided_checkpoint_embeds_exact_frozen_l0_state_and_metric_contract(tmp_path):
    l0, model = RelationLidarModel(), CameraGuidedRelationModel(RelationLidarModel())
    model.l0.load_state_dict(l0.state_dict())
    path = tmp_path / "guided.ckpt"
    save_checkpoint(path, model, epoch=1, global_step=2, sample=_sample(), validation_score=1.0, validation_count=3, validation_metric="camera_guidance_valid_query_weighted_range_mae", source_checkpoints={"l0": {"path": "source.ckpt", "sha256": "x"}})
    payload = torch.load(path, weights_only=False)
    assert payload["validation_metric"] == "camera_guidance_valid_query_weighted_range_mae"
    assert all(torch.equal(value, payload["model"][f"l0.{key}"]) for key, value in l0.state_dict().items())
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            save_checkpoint(tmp_path / f"{bad}.ckpt", model, epoch=1, global_step=2, sample=_sample(), validation_score=bad, validation_count=1)
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "empty.ckpt", model, epoch=1, global_step=2, sample=_sample(), validation_score=1.0, validation_count=1, validation_metric=" ")
