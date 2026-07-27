import numpy as np
import torch

from camera_operator_sr.models.relation import RelationLidarModel
from camera_operator_sr.training.checkpoint import save_checkpoint
from resume_test_support import make_dataset, run


def test_relation_inference_exports_only_relation_fields_and_preserves_observed_rows(tmp_path):
    make_dataset(tmp_path)
    model = RelationLidarModel()
    sample = {"lidar": {"elevation": torch.tensor([-.2, .2]), "azimuth": torch.linspace(-.3, .3, 8)}, "target": {"elevation": torch.tensor([-.2, 0., .2])}}
    checkpoint = tmp_path / "relation.ckpt"; save_checkpoint(checkpoint, model, epoch=0, global_step=0, sample=sample)
    result = tmp_path / "relation.npz"
    run(["scripts/infer.py", "--checkpoint", str(checkpoint), "--frame-root", str(tmp_path / "00" / "000000"), "--output", str(result), "--device", "cpu"])
    values = np.load(result)
    assert {"range", "prior_weights", "final_weights", "correction"} <= set(values.files)
    assert not {"return_probability", "residual", "anchor_entropy"} & set(values.files)
    observed = np.load(tmp_path / "00" / "000000" / "input_range.npy")
    assert np.array_equal(values["range"][0, 0, [0, 2]], observed)

