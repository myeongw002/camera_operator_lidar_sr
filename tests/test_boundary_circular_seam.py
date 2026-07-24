import torch
from camera_operator_sr.evaluation.boundary_metrics import build_range_boundary_mask

def test_circular_seam_discontinuity_is_detected():
    values=torch.tensor([[[[1.,1.,1.,8.]]]]); valid=torch.ones_like(values)
    assert build_range_boundary_mask(values,valid,dilation_pixels=0)[0,0,0,3]
