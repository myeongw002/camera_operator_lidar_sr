import torch
from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.training.reproducibility import seed_everything

def first_parameter(seed):
    seed_everything(seed,deterministic=True); return next(LidarOperatorStudent(lidar_feature_dim=16,hidden_dim=24).parameters()).detach().clone()
def test_seed_controls_model_initialization():
    assert torch.equal(first_parameter(42),first_parameter(42))
    assert not torch.equal(first_parameter(42),first_parameter(43))
