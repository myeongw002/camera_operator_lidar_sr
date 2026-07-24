import torch

from camera_operator_sr.training.modules import DistillModule
from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.models.teacher import CameraGuidedOperatorTeacher


def _batch():
    b, h_in, h_target, width = 2, 4, 8, 16
    input_elevation = torch.tensor([-0.3, -0.1, 0.1, 0.3]).expand(b, -1).clone()
    target_elevation = torch.linspace(-0.3, 0.3, h_target).expand(b, -1).clone()
    azimuth = (torch.arange(width) * (2 * torch.pi / width) - torch.pi + torch.pi / width).expand(b, -1).clone()
    ranges = torch.rand(b, 1, h_in, width) * 20 + 2
    valid = torch.ones_like(ranges)
    valid[:, :, :, [15, 0, 1]] = 0  # queries at c=0 have all-invalid candidates
    target_range = torch.rand(b, 1, h_target, width) * 20 + 2
    target_valid = (torch.rand(b, 1, h_target, width) > 0.3).float()
    target_range *= target_valid
    K = torch.tensor([[[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]]).expand(b, -1, -1).clone()
    T = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]).expand(b, -1, -1).clone()
    return {"lidar": {"range": ranges, "intensity": torch.zeros_like(ranges), "valid": valid, "elevation": input_elevation, "azimuth": azimuth}, "target": {"range": target_range, "valid": target_valid, "elevation": target_elevation}, "camera": {"relative_depth": torch.rand(b, 1, 16, 16), "depth_valid": torch.ones(b, 1, 16, 16)}, "calibration": {"K": K, "T_cam_lidar": T}}


def test_distillation_forward_backward_is_finite():
    torch.manual_seed(2)
    student, teacher, baseline = LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24), CameraGuidedOperatorTeacher(lidar_feature_dim=16, depth_feature_dim=8, hidden_dim=24), LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24)
    module = DistillModule(student, teacher, baseline)
    losses = module(_batch())
    for name, value in losses.items():
        assert torch.isfinite(value), name
    losses["loss"].backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in baseline.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in student.parameters() if parameter.grad is not None)
