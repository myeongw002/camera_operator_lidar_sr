import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

def _data(root):
    frame=root/'00'/'000000'; frame.mkdir(parents=True); w=8
    inp=np.ones((2,w),np.float32)*10; target=np.ones((3,w),np.float32)*10
    for n,v in (('input_range.npy',inp),('input_intensity.npy',np.zeros_like(inp)),('input_valid.npy',np.ones_like(inp)),('target_range.npy',target),('target_valid.npy',np.ones_like(target)),('relative_depth.npy',np.ones((16,16),np.float32)),('depth_valid.npy',np.ones((16,16),np.float32))): np.save(frame/n,v)
    K=np.array([[10,0,8],[0,10,8],[0,0,1]],np.float32); T=np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]],np.float32)
    np.savez(frame/'meta.npz',input_elevation=np.array([-.2,.2],np.float32),target_elevation=np.array([-.2,0,.2],np.float32),azimuth=np.linspace(-.3,.3,w,dtype=np.float32),K=K,T_cam_lidar=T,image_size=np.array([16,16]))
    for n in ('train.txt','val.txt'): (root/n).write_text('00/000000\n')

def _run(args): return subprocess.run([sys.executable,*args],cwd='.',text=True,capture_output=True)
def _check(root,name,kind,depth,adv=None):
    exp=root/'outputs'/name/'seed_42'; assert (exp/'config.json').exists() and (exp/'manifest.json').exists() and (exp/'metrics.jsonl').exists()
    assert (exp/'checkpoints'/'best.ckpt').exists() and (exp/'checkpoints'/'last.ckpt').exists()
    manifest=json.loads((exp/'manifest.json').read_text()); ckpt=torch.load(exp/'checkpoints'/'last.ckpt',weights_only=False); metric=json.loads((exp/'metrics.jsonl').read_text().splitlines()[-1])
    assert manifest['experiment_type']==kind and manifest['depth_mode']==depth and ckpt['experiment_name']==name and ckpt['seed']==42
    assert ckpt['split_hashes']['train']==manifest['splits']['train']['sha256'] and ckpt['validation_score']==metric['validation_score']
    if adv: assert ckpt['advantage_config']==manifest['advantage_config']==adv
    return exp

def test_student_teacher_distillation_write_isolated_reproducible_outputs(tmp_path):
    _data(tmp_path); common=['--dataset-root',str(tmp_path),'--train-split',str(tmp_path/'train.txt'),'--val-split',str(tmp_path/'val.txt'),'--output-root',str(tmp_path/'outputs'),'--seed','42','--epochs','1','--device','cpu']
    student=_run(['scripts/train_student.py','--experiment-name','student_baseline',*common]); assert student.returncode==0,student.stderr; assert '[student] epoch 1/1' in student.stdout
    sdir=_check(tmp_path,'student_baseline','student','none'); before=hashlib.sha256((sdir/'checkpoints'/'last.ckpt').read_bytes()).hexdigest()
    teacher=_run(['scripts/train_teacher.py','--experiment-name','teacher_correct','--baseline-checkpoint',str(sdir/'checkpoints'/'last.ckpt'),'--depth-mode','correct',*common]); assert teacher.returncode==0,teacher.stderr; assert '[teacher:correct] epoch 1/1' in teacher.stdout
    tdir=_check(tmp_path,'teacher_correct','teacher','correct'); adv={'mode':'soft','range_margin':.1,'range_temperature':.1,'return_margin':.05,'return_temperature':.1}
    distill=_run(['scripts/train_distill.py','--experiment-name','distill_correct_soft','--baseline',str(sdir/'checkpoints'/'last.ckpt'),'--teacher',str(tdir/'checkpoints'/'last.ckpt'),'--depth-mode','correct','--advantage-mode','soft',*common]); assert distill.returncode==0,distill.stderr; assert '[distill:correct] epoch 1/1' in distill.stdout
    ddir=_check(tmp_path,'distill_correct_soft','distillation','correct',adv)
    assert len({sdir,tdir,ddir})==3 and hashlib.sha256((sdir/'checkpoints'/'last.ckpt').read_bytes()).hexdigest()==before
