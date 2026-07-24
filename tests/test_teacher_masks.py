import torch

from camera_operator_sr.geometry.visibility import build_camera_query_frustum_mask, build_gt_visible_valid_mask
from camera_operator_sr.losses.supervised import return_loss


def _camera_transform():
    # LiDAR x-forward/y-left/z-up -> camera z-forward/x-right/y-down for this synthetic case.
    return torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]])


def test_teacher_range_and_return_masks_are_separate():
    elevation, azimuth = torch.tensor([0.0]), torch.tensor([0.0, torch.pi])
    K = torch.tensor([[[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]])
    target_xyz = torch.tensor([[[[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    target_valid = torch.tensor([[[[1.0, 0.0]]]])
    visible = build_gt_visible_valid_mask(target_xyz, target_valid, K, _camera_transform(), (10, 10))
    frustum = build_camera_query_frustum_mask(elevation, azimuth, K, _camera_transform(), (10, 10), torch.tensor([2.0, 10.0]))
    assert torch.all(visible <= target_valid)
    assert visible[0, 0, 0, 0] == 1
    assert frustum[0, 0, 0, 0] == 1
    assert frustum[0, 0, 0, 1] == 0

    # The first query is deliberately a valid camera-frustum negative return.
    negative_target = torch.tensor([[[[0.0, 0.0]]]])
    logits = torch.tensor([[[[0.2, -0.3]]]], requires_grad=True)
    loss = return_loss(logits, negative_target, frustum)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_return_loss_handles_all_positive_and_all_negative_targets():
    logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
    mask = torch.ones_like(logits)
    for target in (torch.zeros_like(logits), torch.ones_like(logits)):
        loss = return_loss(logits, target, mask)
        assert torch.isfinite(loss)
