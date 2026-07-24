import torch

from camera_operator_sr.data.masks import build_generated_row_mask
from camera_operator_sr.losses.supervised import supervised_range_loss


def test_observed_rows_do_not_contribute_to_range_loss():
    generated = build_generated_row_mask(torch.tensor([0.0]), torch.tensor([0.0, 1.0]))[None].expand(1, 1, 2, 1)
    prediction = torch.tensor([[[[100.0], [3.0]]]])
    target = torch.tensor([[[[0.0], [1.0]]]])
    valid = torch.ones_like(target)
    loss = supervised_range_loss(prediction, target, valid, generated)
    assert torch.isclose(loss, torch.tensor(1.5))
