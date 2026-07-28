import csv
import json

import torch

from camera_operator_sr.models.relation import CameraGuidedRelationModel, RelationLidarModel
from camera_operator_sr.training.checkpoint import save_checkpoint
from resume_test_support import make_dataset, run


def test_guided_evaluator_writes_common_mask_metrics_and_required_artifacts(tmp_path):
    make_dataset(tmp_path)
    sample = {"lidar": {"elevation": torch.tensor([-.2, .2]), "azimuth": torch.linspace(-.3, .3, 8)}, "target": {"elevation": torch.tensor([-.2, 0., .2])}}
    checkpoint = tmp_path / "guided.ckpt"
    save_checkpoint(checkpoint, CameraGuidedRelationModel(RelationLidarModel()), epoch=0, global_step=0, sample=sample, source_checkpoints={"l0": {"path": "l0.ckpt", "sha256": "test"}})
    output = tmp_path / "evaluation"
    run(["scripts/evaluate_relation_guided.py", "--checkpoint", str(checkpoint), "--dataset-root", str(tmp_path), "--split-file", str(tmp_path / "val.txt"), "--output-root", str(output), "--device", "cpu"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["metadata"]["model_type"] == "relation_guided"
    assert summary["full_generated_supported"]["l0_range_mae"] == summary["full_generated_supported"]["guided_range_mae"]
    assert summary["camera_guidance_valid"]["camera_guidance_count"] > 0
    assert all((output / name).exists() for name in ("region_metrics.csv", "beam_metrics.csv", "distance_metrics.csv", "guided_relation_metrics.csv"))
    assert "guided_over_l0_mae_improvement" in next(csv.reader((output / "guided_relation_metrics.csv").open()))
