import json, subprocess, sys
from pathlib import Path
import numpy as np

def _data(root):
    frame=root/'00'/'000000'; frame.mkdir(parents=True); inp=np.ones((2,8),np.float32)*10; target=np.ones((3,8),np.float32)*10
    for n,v in (('input_range.npy',inp),('input_intensity.npy',np.zeros_like(inp)),('input_valid.npy',np.ones_like(inp)),('target_range.npy',target),('target_valid.npy',np.ones_like(target))): np.save(frame/n,v)
    np.savez(frame/'meta.npz',input_elevation=np.array([-.2,.2],np.float32),target_elevation=np.array([-.2,0,.2],np.float32),azimuth=np.linspace(-.3,.3,8,dtype=np.float32),K=np.eye(3,dtype=np.float32),T_cam_lidar=np.eye(4,dtype=np.float32),image_size=np.array([0,0]))
    for n in ('train.txt','val.txt'): (root/n).write_text('00/000000\n')

def run(args): return subprocess.run([sys.executable,*args],cwd='.',text=True,capture_output=True)
def test_student_resume_overwrite_and_split_hash_protection(tmp_path):
    _data(tmp_path); base=['scripts/train_student.py','--dataset-root',str(tmp_path),'--train-split',str(tmp_path/'train.txt'),'--val-split',str(tmp_path/'val.txt'),'--output-root',str(tmp_path/'out'),'--experiment-name','student','--seed','9','--device','cpu']
    first=run([*base,'--epochs','1']); assert first.returncode==0,first.stderr
    exp=tmp_path/'out'/'student'/'seed_9'; lines=(exp/'metrics.jsonl').read_text().splitlines()
    duplicate=run([*base,'--epochs','1']); assert duplicate.returncode!=0 and 'already exists' in duplicate.stderr
    resumed=run([*base,'--epochs','2','--resume']); assert resumed.returncode==0,resumed.stderr
    after=(exp/'metrics.jsonl').read_text().splitlines(); assert len(after)==len(lines)+1 and json.loads(after[-1])['epoch']==1
    (tmp_path/'train.txt').write_text('00/000000\n# altered\n'); mismatch=run([*base,'--epochs','3','--resume']); assert mismatch.returncode!=0 and 'splits' in mismatch.stderr.lower()
    overwritten=run([*base,'--epochs','1','--overwrite']); assert overwritten.returncode==0 and 'overwriting' in overwritten.stderr.lower()
