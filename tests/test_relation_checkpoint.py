import torch
import pytest

from camera_operator_sr.models.factory import build_model
from camera_operator_sr.models.relation import RelationLidarModel
from camera_operator_sr.training.checkpoint import save_checkpoint, validate_checkpoint_geometry


def sample():
    return {"lidar": {"elevation": torch.tensor([-1., 1.]), "azimuth": torch.tensor([-1., 1.])}, "target": {"elevation": torch.tensor([-1., 0., 1.])}}


def test_relation_checkpoint_uses_schema_four_and_candidate_contract(tmp_path):
    model = RelationLidarModel(); path = tmp_path / "l0.ckpt"; save_checkpoint(path, model, epoch=0, global_step=0, sample=sample())
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["checkpoint_schema_version"] == 4
    assert checkpoint["geometry"]["anchor_slots"] == [1, 4]
    restored = build_model(checkpoint["model_config"]); restored.load_state_dict(checkpoint["model"])
    validate_checkpoint_geometry(checkpoint, sample())


def test_relation_checkpoint_rejects_every_l0_geometry_contract_mismatch(tmp_path):
    model = RelationLidarModel(); path = tmp_path / "l0.ckpt"; save_checkpoint(path, model, epoch=0, global_step=0, sample=sample())
    checkpoint = torch.load(path, weights_only=False)
    for key, value in (("input_elevation", [-.9, 1.]), ("target_elevation", [-1., .1, 1.]), ("azimuth", [-1., .5]), ("width", 3), ("candidate_horizontal_radius", 2), ("candidate_count", 5), ("candidate_layout", "bad"), ("anchor_slots", [0, 4])):
        altered = {**checkpoint, "geometry": dict(checkpoint["geometry"])}
        altered["geometry"][key] = value
        if key == "candidate_horizontal_radius":
            altered["geometry"]["candidate_count"] = 10
        with pytest.raises(ValueError):
            validate_checkpoint_geometry(altered, sample())
