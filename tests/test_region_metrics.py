import torch

from camera_operator_sr.evaluation.region_metrics import build_region_masks


def test_region_masks_cover_expected_azimuths():
    regions = build_region_masks(torch.tensor([0.0, torch.pi / 2, torch.pi]))
    assert regions["full"].sum() == 3
    assert regions["rear"][0, 0, 0, 2]
    assert regions["camera_frustum"][0, 0, 0, 0]
