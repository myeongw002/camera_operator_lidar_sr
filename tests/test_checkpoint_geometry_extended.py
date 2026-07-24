import pytest
import torch

from camera_operator_sr.training.checkpoint import build_geometry_metadata, extract_dataset_geometry, save_checkpoint, validate_checkpoint_geometry
from camera_operator_sr.models.student import LidarOperatorStudent


def sample(width=4):
    return {"lidar": {"elevation": torch.tensor([-0.2, 0.2]), "azimuth": torch.linspace(-1, 1, width)}, "target": {"elevation": torch.tensor([-0.2, 0.0, 0.2])}}


def checkpoint(tmp_path):
    model=LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24, horizontal_radius=1); path=tmp_path/"model.ckpt"
    save_checkpoint(path,model,epoch=0,global_step=0,sample=sample()); return torch.load(path,weights_only=False)


def test_geometry_metadata_roundtrip_and_all_mismatches(tmp_path):
    ckpt=checkpoint(tmp_path); validate_checkpoint_geometry(ckpt,sample())
    geometry=ckpt["geometry"]
    assert geometry["width"] == len(geometry["azimuth"]) == 4
    assert geometry["input_beam_count"] == len(geometry["input_elevation"])
    assert geometry["target_beam_count"] == len(geometry["target_elevation"])
    variants=[]
    changed=sample(); changed["lidar"]["elevation"][0]+=0.01; variants.append(changed)
    changed=sample(); changed["lidar"]["elevation"]=torch.tensor([-0.2,0.0,0.2]); variants.append(changed)
    changed=sample(); changed["target"]["elevation"][0]+=0.01; variants.append(changed)
    changed=sample(); changed["target"]["elevation"]=torch.tensor([-0.2,0.2]); variants.append(changed)
    changed=sample(); changed["lidar"]["azimuth"]=changed["lidar"]["azimuth"].roll(1); variants.append(changed)
    changed=sample(); changed["lidar"]["azimuth"]=changed["lidar"]["azimuth"].flip(0); variants.append(changed)
    variants.append(sample(width=2))
    for value in variants:
        with pytest.raises(ValueError): validate_checkpoint_geometry(ckpt,value)
    altered=dict(geometry); altered["candidate_horizontal_radius"]=2; altered["candidate_count"]=10
    with pytest.raises(ValueError): validate_checkpoint_geometry(ckpt,altered)
    altered=dict(geometry); altered["candidate_count"]=5
    with pytest.raises(ValueError): validate_checkpoint_geometry(ckpt,altered)


def test_metadata_rejects_inconsistent_counts():
    with pytest.raises(ValueError): build_geometry_metadata(input_elevation=torch.tensor([0.]),target_elevation=torch.tensor([0.]),azimuth=torch.tensor([0.]),candidate_horizontal_radius=1,candidate_count=1)
