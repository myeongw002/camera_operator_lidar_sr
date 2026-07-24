import torch

from camera_operator_sr.losses.advantage_mask import compute_return_advantage


def test_advantage_modes_gate_bad_teacher():
    baseline, teacher, target = torch.tensor([[[[2.0]]]]), torch.tensor([[[[-2.0]]]]), torch.zeros(1, 1, 1, 1)
    assert compute_return_advantage(baseline, teacher, target, mode="hard").item() == 1
    assert compute_return_advantage(baseline, teacher, target, mode="none").item() == 1
    bad = compute_return_advantage(teacher, baseline, target, mode="hard")
    assert bad.item() == 0
