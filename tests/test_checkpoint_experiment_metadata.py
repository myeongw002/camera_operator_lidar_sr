import torch
from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.training.checkpoint import save_checkpoint

def test_checkpoint_contains_experiment_metadata_and_advantage(tmp_path):
    sample={'lidar':{'elevation':torch.tensor([-.2,.2]),'azimuth':torch.tensor([-1.,1.])},'target':{'elevation':torch.tensor([-.2,0.,.2])}}
    manifest={'experiment_name':'distill','splits':{'train':{'sha256':'a'},'validation':None,'test':None}}
    metadata={'experiment_name':'distill','experiment_type':'distillation','seed':7,'deterministic':True,'run_config':{'seed':7},'manifest':manifest}
    path=tmp_path/'x.ckpt'; save_checkpoint(path,LidarOperatorStudent(lidar_feature_dim=16,hidden_dim=24),epoch=1,global_step=2,sample=sample,validation_score=.2,validation_count=3,experiment_metadata=metadata,advantage_config={'mode':'soft'})
    c=torch.load(path,weights_only=False)
    for key in ('experiment_name','experiment_type','seed','deterministic','run_config','manifest','split_hashes','geometry','validation_score'): assert key in c
    assert c['advantage_config']=={'mode':'soft'} and c['split_hashes']['train']=='a'
