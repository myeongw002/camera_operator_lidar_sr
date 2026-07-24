import torch
from camera_operator_sr.evaluation.boundary_metrics import build_range_boundary_mask
from camera_operator_sr.evaluation.region_metrics import build_region_masks

def test_boundary_and_interior_partition_visible_region():
    ranges=torch.tensor([[[[2.,2.,8.,8.]]]]); valid=torch.ones_like(ranges)
    boundary=build_range_boundary_mask(ranges,valid,dilation_pixels=0)
    assert boundary.any()
    regions=build_region_masks(torch.arange(4.),gt_camera_visible=valid,camera_boundary=boundary)
    assert not (regions["camera_boundary"] & regions["camera_interior"]).any()
    assert torch.equal(regions["camera_boundary"] | regions["camera_interior"],valid.bool())
