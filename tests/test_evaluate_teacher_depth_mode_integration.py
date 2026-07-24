import csv
import json
import subprocess
import sys

import numpy as np
import torch

from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.models.teacher import CameraGuidedOperatorTeacher
from camera_operator_sr.training.checkpoint import save_checkpoint


def _dataset(root):
    frame=root/"00"/"000000"; frame.mkdir(parents=True)
    input_range=np.ones((2,8),np.float32)*10; target=np.ones((3,8),np.float32)*10
    for name,value in (("input_range.npy",input_range),("input_intensity.npy",np.zeros_like(input_range)),("input_valid.npy",np.ones_like(input_range)),("target_range.npy",target),("target_valid.npy",np.ones_like(target)),("relative_depth.npy",np.ones((4,4),np.float32)),("depth_valid.npy",np.ones((4,4),np.float32))): np.save(frame/name,value)
    np.savez(frame/"meta.npz",input_elevation=np.array([-.2,.2],np.float32),target_elevation=np.array([-.2,0,.2],np.float32),azimuth=np.linspace(-1,1,8,dtype=np.float32),K=np.eye(3,dtype=np.float32),T_cam_lidar=np.eye(4,dtype=np.float32),image_size=np.array([0,0]))
    split=root/"test.txt"; split.write_text("00/000000\n"); return split


def _checkpoint(root,name,model,mode):
    sample={"lidar":{"elevation":torch.tensor([-.2,.2]),"azimuth":torch.linspace(-1,1,8)},"target":{"elevation":torch.tensor([-.2,0,.2])}}
    path=root/name; save_checkpoint(path,model,epoch=0,global_step=0,sample=sample,depth_mode=mode); return path


def _run(root, baseline, teacher, *extra):
    return subprocess.run([sys.executable,"scripts/evaluate_teacher.py","--baseline-checkpoint",str(baseline),"--teacher-correct",str(teacher),"--dataset-root",str(root),"--split-file",str(root/"test.txt"),"--output-root",str(root/"out"),"--device","cpu",*extra],cwd=".",text=True,capture_output=True)


def test_teacher_evaluation_slot_depth_modes_are_enforced_and_recorded(tmp_path):
    split=_dataset(tmp_path); baseline=_checkpoint(tmp_path,"baseline.ckpt",LidarOperatorStudent(lidar_feature_dim=16,hidden_dim=24),"none")
    correct=_checkpoint(tmp_path,"correct.ckpt",CameraGuidedOperatorTeacher(lidar_feature_dim=16,hidden_dim=24),"correct")
    none=_checkpoint(tmp_path,"none.ckpt",CameraGuidedOperatorTeacher(lidar_feature_dim=16,hidden_dim=24),"none")
    normal=_run(tmp_path,baseline,correct); assert normal.returncode == 0, normal.stderr
    rows=list(csv.DictReader((tmp_path/"out"/"teacher_comparison.csv").open())); row=next(value for value in rows if value["model"]=="teacher_correct")
    assert row["checkpoint_depth_mode"] == row["evaluation_depth_mode"] == "correct"
    metadata=json.loads((tmp_path/"out"/"teacher_superiority.json").read_text())
    assert metadata["models"]["teacher_correct"] == {"checkpoint_depth_mode":"correct","evaluation_depth_mode":"correct"}
    bad=_run(tmp_path,baseline,none); assert bad.returncode != 0
    assert all(word in bad.stderr.lower() for word in ("depth mode mismatch","teacher_correct","none","correct"))
    allowed=_run(tmp_path,baseline,none,"--allow-depth-mode-mismatch"); assert allowed.returncode == 0
    assert "explicitly allowed" in allowed.stderr.lower()
    rows=list(csv.DictReader((tmp_path/"out"/"teacher_comparison.csv").open())); row=next(value for value in rows if value["model"]=="teacher_correct")
    assert row["checkpoint_depth_mode"] == "none" and row["evaluation_depth_mode"] == row["depth_mode"] == "correct"
    missing=torch.load(none,weights_only=False); del missing["depth_mode"]; missing_path=tmp_path/"missing.ckpt"; torch.save(missing,missing_path)
    absent=_run(tmp_path,baseline,missing_path); assert absent.returncode != 0 and "depth_mode" in absent.stderr


def test_other_teacher_slot_is_not_special_cased(tmp_path):
    _dataset(tmp_path); baseline=_checkpoint(tmp_path,"baseline.ckpt",LidarOperatorStudent(lidar_feature_dim=16,hidden_dim=24),"none")
    correct=_checkpoint(tmp_path,"correct.ckpt",CameraGuidedOperatorTeacher(lidar_feature_dim=16,hidden_dim=24),"correct")
    result=subprocess.run([sys.executable,"scripts/evaluate_teacher.py","--baseline-checkpoint",str(baseline),"--teacher-correct",str(correct),"--teacher-none",str(correct),"--dataset-root",str(tmp_path),"--split-file",str(tmp_path/"test.txt"),"--output-root",str(tmp_path/"out"),"--device","cpu"],cwd=".",text=True,capture_output=True)
    assert result.returncode != 0 and "teacher_none" in result.stderr and "depth mode mismatch" in result.stderr.lower()
