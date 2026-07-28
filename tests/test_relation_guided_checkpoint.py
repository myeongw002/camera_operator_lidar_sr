import torch

from camera_operator_sr.models.factory import build_model
from camera_operator_sr.models.relation import CameraGuidedRelationModel, RelationLidarModel
from camera_operator_sr.training.checkpoint import save_checkpoint, validate_checkpoint_geometry


def sample(): return {"lidar": {"elevation": torch.tensor([-1., 1.]), "azimuth": torch.linspace(-1., 1., 4)}, "target": {"elevation": torch.tensor([-1., 0., 1.])}}


def test_guided_schema_four_checkpoint_roundtrips_through_factory(tmp_path):
    model = CameraGuidedRelationModel(RelationLidarModel()); path = tmp_path / "guided.ckpt"
    save_checkpoint(path, model, epoch=0, global_step=0, sample=sample(), source_checkpoints={"l0": {"path": "l0.ckpt", "sha256": "x"}})
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["checkpoint_schema_version"] == 4
    assert checkpoint["model_config"]["model_type"] == "relation_guided"
    assert checkpoint["source_checkpoints"]["l0"]["path"] == "l0.ckpt"
    restored = build_model(checkpoint["model_config"]); restored.load_state_dict(checkpoint["model"])
    validate_checkpoint_geometry(checkpoint, sample())
