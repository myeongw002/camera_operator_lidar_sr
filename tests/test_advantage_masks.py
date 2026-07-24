import torch

from camera_operator_sr.losses.advantage_mask import build_distillation_masks, compute_range_advantage, compute_return_advantage


def test_range_and_return_advantages_use_different_target_support():
    generated = torch.ones(1, 1, 1, 2)
    visible = torch.ones_like(generated)
    frustum = torch.ones_like(generated)
    target_valid = torch.tensor([[[[1.0, 0.0]]]])
    target_range = torch.tensor([[[[10.0, 0.0]]]])
    baseline_range = torch.tensor([[[[20.0, 5.0]]]])
    teacher_range = torch.tensor([[[[10.0, 9.0]]]])
    baseline_logits = torch.tensor([[[[2.0, 2.0]]]])
    teacher_logits = torch.tensor([[[[5.0, -5.0]]]])
    range_advantage = compute_range_advantage(baseline_range, teacher_range, target_range, target_valid)
    return_advantage = compute_return_advantage(baseline_logits, teacher_logits, target_valid)
    masks = build_distillation_masks(generated, visible, frustum, target_valid, range_advantage, return_advantage)
    assert masks.operator[0, 0, 0, 0] > 0 and masks.operator[0, 0, 0, 1] == 0
    assert masks.return_mask[0, 0, 0, 0] > 0 and masks.return_mask[0, 0, 0, 1] > 0
    worse_teacher = compute_return_advantage(baseline_logits, torch.tensor([[[[-5.0, 5.0]]]]), target_valid)
    assert worse_teacher[0, 0, 0, 1] < return_advantage[0, 0, 0, 1]
