import torch

from camera_operator_sr.data.normalization import depth_gradients


def test_depth_gradients_ignore_invalid_boundaries():
    depth = torch.tensor([[[[1.0, 5.0, 1.0], [1.0, 2.0, 3.0]]]])
    valid = torch.tensor([[[[1, 0, 1], [1, 1, 1]]]], dtype=torch.bool)
    gx, gy = depth_gradients(depth, valid)
    assert gx[0, 0, 0, 0] == 0 and gx[0, 0, 0, 1] == 0
    assert gy[0, 0, 0, 1] == 0
    assert gx[0, 0, 1, 0] == 1
    assert gx[0, 0, 1, 2] == 0


def test_constant_valid_depth_has_zero_gradients():
    depth, valid = torch.ones(1, 1, 3, 4), torch.ones(1, 1, 3, 4)
    gx, gy = depth_gradients(depth, valid)
    assert gx.eq(0).all() and gy.eq(0).all()
