import torch
from camera_operator_sr.geometry.visibility import build_gt_visible_valid_mask

def test_zbuffer_visibility_keeps_front_surface_only():
    xyz=torch.tensor([[[[0.,0.,2.],[0.,0.,4.],[10.,0.,2.]]]])
    valid=torch.ones(1,1,1,3); K=torch.tensor([[[1.,0.,1.],[0.,1.,1.],[0.,0.,1.]]]); T=torch.eye(4)[None]
    mask=build_gt_visible_valid_mask(xyz,valid,K,T,(3,3),abs_tolerance=.01)
    assert mask.shape==valid.shape and mask[0,0,0,0] and not mask[0,0,0,1] and not mask[0,0,0,2]
