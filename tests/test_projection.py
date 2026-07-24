import torch

from camera_operator_sr.geometry.projection import project_lidar_points


def test_identity_projection_uses_pixel_centres():
    points = torch.tensor([[[1.0, 2.0, 1.0]]])
    K = torch.eye(3).unsqueeze(0)
    projection = project_lidar_points(points, K, torch.eye(4).unsqueeze(0), (4, 4))
    assert projection.valid.item()
    assert torch.allclose(projection.uv, torch.tensor([[[1.0, 2.0]]]))
    assert torch.allclose(projection.normalized_grid, torch.tensor([[[-0.25, 0.25]]]))
