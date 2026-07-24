import subprocess
import sys

import numpy as np
import torch

from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.models.teacher import CameraGuidedOperatorTeacher
from camera_operator_sr.training.checkpoint import save_checkpoint


def _frame(root, width=8):
    path=root/"00"/"000000"; path.mkdir(parents=True)
    input_range=np.ones((2,width),np.float32)*10; target_range=np.ones((3,width),np.float32)*10
    for name,value in (("input_range.npy",input_range),("input_intensity.npy",np.zeros_like(input_range)),("input_valid.npy",np.ones_like(input_range)),("target_range.npy",target_range),("target_valid.npy",np.ones_like(target_range))): np.save(path/name,value)
    np.save(path/"relative_depth.npy",np.ones((4,4),np.float32)); np.save(path/"depth_valid.npy",np.ones((4,4),np.float32))
    np.savez(path/"meta.npz",input_elevation=np.array([-.2,.2],np.float32),target_elevation=np.array([-.2,0,.2],np.float32),azimuth=np.linspace(-1,1,width,dtype=np.float32),K=np.eye(3,dtype=np.float32),T_cam_lidar=np.eye(4,dtype=np.float32),image_size=np.array([0,0]))
    (root/"test.txt").write_text("00/000000\n"); return path


def _checkpoint(root, name, model, mode):
    sample={"lidar":{"elevation":torch.tensor([-.2,.2]),"azimuth":torch.linspace(-1,1,8)},"target":{"elevation":torch.tensor([-.2,0,.2])}}
    path=root/name; save_checkpoint(path,model,epoch=0,global_step=0,sample=sample,depth_mode=mode); return path


def _run(arguments): return subprocess.run([sys.executable,*arguments],cwd=".",text=True,capture_output=True)


def test_cli_rejects_geometry_and_depth_mode_mismatch(tmp_path):
    frame=_frame(tmp_path); baseline=_checkpoint(tmp_path,"baseline.ckpt",LidarOperatorStudent(lidar_feature_dim=16,hidden_dim=24),"none")
    teacher=_checkpoint(tmp_path,"teacher.ckpt",CameraGuidedOperatorTeacher(lidar_feature_dim=16,hidden_dim=24),"correct")
    good=_run(["scripts/evaluate_sr.py","--checkpoint",str(baseline),"--dataset-root",str(tmp_path),"--split-file",str(tmp_path/"test.txt"),"--output-root",str(tmp_path/"eval"),"--device","cpu"])
    assert good.returncode == 0, good.stderr
    student_train=_run(["scripts/train_student.py","--dataset-root",str(tmp_path),"--epochs","0","--output-root",str(tmp_path/"runs"),"--experiment-name","student","--seed","1","--device","cpu"])
    assert student_train.returncode == 0, student_train.stderr
    teacher_train=_run(["scripts/train_teacher.py","--dataset-root",str(tmp_path),"--baseline-checkpoint",str(baseline),"--depth-mode","none","--epochs","0","--output-root",str(tmp_path/"runs"),"--experiment-name","teacher","--seed","1","--device","cpu"])
    assert teacher_train.returncode == 0, teacher_train.stderr
    teacher_eval=_run(["scripts/evaluate_teacher.py","--baseline-checkpoint",str(baseline),"--teacher-correct",str(teacher),"--dataset-root",str(tmp_path),"--split-file",str(tmp_path/"test.txt"),"--output-root",str(tmp_path/"teacher_eval"),"--device","cpu"])
    assert teacher_eval.returncode == 0, teacher_eval.stderr
    shifted=torch.load(baseline,weights_only=False); shifted["geometry"]["azimuth"]=shifted["geometry"]["azimuth"][1:]+shifted["geometry"]["azimuth"][:1]; shifted_path=tmp_path/"shifted.ckpt"; torch.save(shifted,shifted_path)
    bad=_run(["scripts/evaluate_sr.py","--checkpoint",str(shifted_path),"--dataset-root",str(tmp_path),"--split-file",str(tmp_path/"test.txt"),"--output-root",str(tmp_path/"bad"),"--device","cpu"])
    assert bad.returncode != 0 and "azimuth" in bad.stderr.lower()
    radius=torch.load(baseline,weights_only=False); radius["geometry"]["candidate_horizontal_radius"]=2; radius["geometry"]["candidate_count"]=10; radius_path=tmp_path/"radius.ckpt"; torch.save(radius,radius_path)
    bad=_run(["scripts/evaluate_sr.py","--checkpoint",str(radius_path),"--dataset-root",str(tmp_path),"--split-file",str(tmp_path/"test.txt"),"--output-root",str(tmp_path/"radius_bad"),"--device","cpu"])
    assert bad.returncode != 0 and "candidate_horizontal_radius" in bad.stderr
    width_root=tmp_path/"width"; width_frame=_frame(width_root,width=4)
    bad=_run(["scripts/infer.py","--checkpoint",str(baseline),"--frame-root",str(width_frame),"--output",str(tmp_path/"out.npz"),"--device","cpu"])
    assert bad.returncode != 0 and ("width" in bad.stderr.lower() or "azimuth" in bad.stderr.lower())
    bad=_run(["scripts/train_distill.py","--dataset-root",str(tmp_path),"--teacher",str(teacher),"--baseline",str(baseline),"--depth-mode","none","--epochs","0","--output-root",str(tmp_path/"runs"),"--experiment-name","distill_bad","--seed","1","--device","cpu"])
    assert bad.returncode != 0 and "depth mode mismatch" in bad.stderr.lower()
    allowed=_run(["scripts/train_distill.py","--dataset-root",str(tmp_path),"--teacher",str(teacher),"--baseline",str(baseline),"--depth-mode","none","--allow-depth-mode-mismatch","--epochs","0","--output-root",str(tmp_path/"runs"),"--experiment-name","distill_allowed","--seed","1","--device","cpu"])
    assert allowed.returncode == 0 and "explicitly allowed" in allowed.stderr.lower()
