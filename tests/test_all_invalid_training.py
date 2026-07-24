import torch

from camera_operator_sr.losses.advantage_mask import build_distillation_masks
from camera_operator_sr.losses.supervised import return_loss, supervised_range_loss


def test_all_invalid_is_excluded_from_train_masks_but_not_final_return_eval():
    generated = torch.ones(1, 1, 1, 2)
    has_candidate = torch.tensor([[[[1.0, 0.0]]]])
    target = torch.tensor([[[[2.0, 7.0]]]])
    valid = torch.ones_like(target)
    prediction = torch.tensor([[[[3.0, 0.0]]]], requires_grad=True)
    assert supervised_range_loss(prediction, target, valid, generated * has_candidate).item() == 0.5
    logits = torch.zeros_like(prediction, requires_grad=True)
    assert torch.isfinite(return_loss(logits, valid, generated * has_candidate))
    masks = build_distillation_masks(generated, generated, generated, valid, torch.ones_like(target), torch.ones_like(target), has_candidate)
    assert masks.operator[0, 0, 0, 1] == 0 and masks.return_mask[0, 0, 0, 1] == 0
