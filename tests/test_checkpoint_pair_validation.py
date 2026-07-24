import copy
import pytest
import torch

from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.models.teacher import CameraGuidedOperatorTeacher
from camera_operator_sr.training.checkpoint import save_checkpoint, validate_checkpoint_pair


def _sample(): return {"lidar":{"elevation":torch.tensor([-0.2,0.2]),"azimuth":torch.linspace(-1,1,4)},"target":{"elevation":torch.tensor([-0.2,0.,0.2])}}
def _checkpoint(tmp_path, name, cls):
    model=cls(lidar_feature_dim=16,hidden_dim=24); path=tmp_path/name; save_checkpoint(path,model,epoch=0,global_step=0,sample=_sample(),depth_mode="correct" if cls is CameraGuidedOperatorTeacher else "none"); return torch.load(path,weights_only=False)


def test_pair_rejects_geometry_candidate_and_shared_model_mismatches(tmp_path):
    baseline=_checkpoint(tmp_path,"baseline.ckpt",LidarOperatorStudent); teacher=_checkpoint(tmp_path,"teacher.ckpt",CameraGuidedOperatorTeacher)
    validate_checkpoint_pair(teacher,baseline,left_name="teacher",right_name="baseline")
    for key, value in (("target_elevation",[-.2,.1,.2]),("azimuth",[-1.,-.3,.3,1.]),("width",8),("candidate_horizontal_radius",2),("candidate_count",10)):
        changed=copy.deepcopy(teacher); changed["geometry"][key]=value
        if key=="width": changed["geometry"]["azimuth"]=[-1.]*8
        if key=="candidate_horizontal_radius": changed["geometry"]["candidate_count"]=10
        with pytest.raises(ValueError): validate_checkpoint_pair(changed,baseline,left_name="teacher",right_name="baseline")
    changed=copy.deepcopy(teacher); changed["model_config"]["hidden_dim"]=99
    with pytest.raises(ValueError): validate_checkpoint_pair(changed,baseline,left_name="teacher",right_name="baseline")
