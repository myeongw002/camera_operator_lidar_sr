import torch

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
