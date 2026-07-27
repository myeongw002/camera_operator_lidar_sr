import csv
import json

import torch

from camera_operator_sr.models.relation import RelationLidarModel
from camera_operator_sr.training.checkpoint import save_checkpoint
from resume_test_support import make_dataset, run


def test_relation_evaluator_compares_zero_initialized_prior_and_final_on_one_mask(tmp_path):
    make_dataset(tmp_path)
    model = RelationLidarModel()
    sample = {"lidar": {"elevation": torch.tensor([-.2, .2]), "azimuth": torch.linspace(-.3, .3, 8)}, "target": {"elevation": torch.tensor([-.2, 0., .2])}}
    checkpoint = tmp_path / "relation.ckpt"
    save_checkpoint(checkpoint, model, epoch=0, global_step=0, sample=sample)
    output = tmp_path / "evaluation"
    run(["scripts/evaluate_relation.py", "--checkpoint", str(checkpoint), "--dataset-root", str(tmp_path), "--split-file", str(tmp_path / "val.txt"), "--output-root", str(output), "--device", "cpu", "--distance-bins", "0", "20", "inf"])
    summary = json.loads((output / "summary.json").read_text())
    global_metrics = summary["global"]
    assert summary["metadata"]["model_type"] == "relation_l0"
    assert global_metrics["prior_range_mae"] == global_metrics["final_range_mae"]
    assert global_metrics["prior_range_rmse"] == global_metrics["final_range_rmse"]
    assert global_metrics["mae_improvement"] == global_metrics["rmse_improvement"] == global_metrics["mean_abs_correction"] == 0.0
    assert global_metrics["supported_count"] == global_metrics["valid_target_count"] == 8
    assert global_metrics["anchor_coverage"] == 1.0
    assert "both_anchor_count" in global_metrics
    assert all((output / name).exists() for name in ("region_metrics.csv", "beam_metrics.csv", "distance_metrics.csv", "relation_metrics.csv"))
    rows = list(csv.DictReader((output / "beam_metrics.csv").open()))
    assert "both_anchor_count" in rows[0]
    for name in ("region_metrics.csv", "distance_metrics.csv", "relation_metrics.csv"):
        assert "both_anchor_count" in next(csv.reader((output / name).open()))
    assert next(row for row in rows if row["target_row"] == "0")["empty_group"] == "True"
