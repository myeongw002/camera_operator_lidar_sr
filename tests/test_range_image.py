import torch

from camera_operator_sr.data.range_image import pointcloud_to_range_image, range_image_to_pointcloud


def test_nearest_collision_and_round_trip():
    elevation = torch.tensor([0.0])
    points = torch.tensor([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    image = pointcloud_to_range_image(points, None, elevation, width=4)
    assert image.valid.sum() == 1
    assert image.range[0, 0, 2] == 1.0
    restored = range_image_to_pointcloud(image.range, image.valid, elevation, torch.tensor([-2.3561945, -0.7853982, 0.7853982, 2.3561945]))
    assert torch.allclose(restored[0, 0, 2], torch.tensor([0.7071068, 0.7071068, 0.0]), atol=1e-5)
