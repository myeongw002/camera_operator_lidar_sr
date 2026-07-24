import pytest
import torch

from camera_operator_sr.losses.distillation import operator_kl_loss


@pytest.mark.parametrize("valid", [
    torch.tensor([[[[1, 1, 1], [1, 1, 1]]]], dtype=torch.bool),
    torch.tensor([[[[1, 0, 1], [0, 1, 0]]]], dtype=torch.bool),
    torch.tensor([[[[0, 1, 0], [0, 0, 1]]]], dtype=torch.bool),
    torch.tensor([[[[0, 0, 0], [0, 0, 0]]]], dtype=torch.bool),
])
def test_operator_kl_is_finite_for_every_validity_pattern(valid):
    student = torch.randn(*valid.shape, requires_grad=True)
    teacher = torch.randn(*valid.shape, requires_grad=True)
    mask = torch.ones(valid.shape[0], 1, valid.shape[1], valid.shape[2])
    loss = operator_kl_loss(student, teacher, valid, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_operator_kl_is_zero_for_equal_logits_and_excludes_all_invalid_queries():
    logits = torch.randn(1, 1, 3, 3, requires_grad=True)
    valid = torch.tensor([[[[1, 0, 1], [0, 0, 0], [0, 1, 0]]]], dtype=torch.bool)
    loss = operator_kl_loss(logits, logits.detach(), valid, torch.ones(1, 1, 1, 3))
    assert loss.item() < 1e-6
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_operator_kl_is_finite_under_autocast():
    device = torch.device("cuda")
    student = torch.randn(1, 2, 3, 4, device=device, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    valid = torch.tensor([[[[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]]], device=device, dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = operator_kl_loss(student, teacher, valid, torch.ones(1, 1, 2, 3, device=device))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all() and teacher.grad is None
