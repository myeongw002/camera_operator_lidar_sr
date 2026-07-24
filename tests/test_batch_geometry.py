import pytest
import torch

from camera_operator_sr.geometry.validation import assert_shared_geometry


def test_batch_geometry_rejects_different_beam_patterns():
    with pytest.raises(ValueError, match="share"):
        assert_shared_geometry(torch.tensor([[0.0, 1.0], [0.0, 1.1]]), "elevation")


def test_batch_geometry_accepts_shared_patterns():
    assert_shared_geometry(torch.tensor([[0.0, 1.0], [0.0, 1.0]]), "elevation")
