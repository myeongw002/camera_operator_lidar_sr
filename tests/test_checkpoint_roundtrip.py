import torch

from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.models.teacher import CameraGuidedOperatorTeacher
from camera_operator_sr.training.checkpoint import save_checkpoint, validate_checkpoint_geometry


def test_checkpoint_preserves_model_and_geometry(tmp_path):
    model = LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24)
    sample = {"lidar": {"elevation": torch.tensor([-0.2, 0.2]), "azimuth": torch.tensor([-1.0, 1.0])}, "target": {"elevation": torch.tensor([-0.2, 0.0, 0.2])}}
    path = tmp_path / "last.ckpt"
    save_checkpoint(path, model, epoch=1, global_step=4, sample=sample)
    checkpoint = torch.load(path, weights_only=False)
    restored = LidarOperatorStudent(**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["model"])
    validate_checkpoint_geometry(checkpoint, sample)
    assert checkpoint["checkpoint_schema_version"] == 3
    assert checkpoint["geometry"]["width"] == 2
    changed = {"lidar": {"elevation": torch.tensor([-0.2, 0.3]), "azimuth": torch.tensor([-1.0, 1.0])}, "target": sample["target"]}
    try:
        validate_checkpoint_geometry(checkpoint, changed)
    except ValueError:
        pass
    else:
        raise AssertionError("different beam pattern must be rejected")


def test_student_teacher_and_distillation_checkpoint_schema_roundtrip(tmp_path):
    sample = {"lidar": {"elevation": torch.tensor([-0.2, 0.2]), "azimuth": torch.tensor([-1.0, 1.0])}, "target": {"elevation": torch.tensor([-0.2, 0.0, 0.2])}}
    for name, model, mode in (("student", LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24), "none"), ("teacher", CameraGuidedOperatorTeacher(lidar_feature_dim=16, hidden_dim=24), "correct"), ("distillation", LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24), "correct")):
        path = tmp_path / f"{name}.ckpt"
        save_checkpoint(path, model, epoch=1, global_step=2, sample=sample, depth_mode=mode, validation_score=0.5, validation_count=3)
        checkpoint = torch.load(path, weights_only=False)
        for key in ("checkpoint_schema_version", "geometry", "model_config", "validation_metric", "validation_score", "validation_count"):
            assert key in checkpoint
        assert checkpoint["validation_metric"] == "global_query_weighted_range_mae"
        assert isinstance(checkpoint["validation_score"], float) and isinstance(checkpoint["validation_count"], int)
        for key in ("input_elevation", "target_elevation", "azimuth", "width", "input_beam_count", "target_beam_count", "candidate_horizontal_radius", "candidate_count"):
            assert key in checkpoint["geometry"]
        assert checkpoint["geometry"]["width"] == len(checkpoint["geometry"]["azimuth"])
        if name == "teacher": assert checkpoint["depth_mode"] == "correct"
