import pytest
import torch

from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.training.checkpoint import save_checkpoint


def _sample():
    return {"lidar": {"elevation": torch.tensor([-0.2, 0.2]), "azimuth": torch.tensor([-1.0, 1.0])}, "target": {"elevation": torch.tensor([-0.2, 0.0, 0.2])}}


def test_best_checkpoint_contains_global_validation_metadata(tmp_path):
    path = tmp_path / "best.ckpt"
    save_checkpoint(path, LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24), epoch=1, global_step=2, sample=_sample(), validation_score=110 / 101, validation_count=101)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["validation_metric"] == "global_query_weighted_range_mae"
    assert isinstance(checkpoint["validation_score"], float)
    assert isinstance(checkpoint["validation_count"], int)


def test_best_checkpoint_rejects_zero_validation_count(tmp_path):
    with pytest.raises(ValueError, match="validation_count"):
        save_checkpoint(tmp_path / "best.ckpt", LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24), epoch=1, global_step=2, sample=_sample(), validation_score=1.0, validation_count=0)
