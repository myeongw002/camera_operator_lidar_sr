import csv
import math
import subprocess
import sys

import numpy as np
import torch

from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import ProcessedTrainingDataset
from camera_operator_sr.data.masks import build_generated_row_mask
from camera_operator_sr.training.checkpoint import save_checkpoint


def _write_frame(root, sequence, frame):
    path = root / sequence / frame
    path.mkdir(parents=True)
    input_range = np.full((2, 8), 10.0, dtype=np.float32)
    target_range = np.stack((np.full(8, 5.0), np.full(8, 15.0), np.full(8, 25.0))).astype(np.float32)
    np.save(path / "input_range.npy", input_range)
    np.save(path / "input_intensity.npy", np.zeros_like(input_range))
    np.save(path / "input_valid.npy", np.ones_like(input_range))
    np.save(path / "target_range.npy", target_range)
    np.save(path / "target_valid.npy", np.ones_like(target_range))
    np.savez(path / "meta.npz", input_elevation=np.array([-0.2, 0.2], dtype=np.float32), target_elevation=np.array([-0.2, 0.0, 0.2], dtype=np.float32), azimuth=np.linspace(-1.0, 1.0, 8, dtype=np.float32), K=np.eye(3, dtype=np.float32), T_cam_lidar=np.eye(4, dtype=np.float32), image_size=np.array([0, 0]))


def test_evaluate_cli_writes_real_detailed_csvs(tmp_path):
    _write_frame(tmp_path, "00", "000000")
    split = tmp_path / "test.txt"; split.write_text("00/000000\n")
    model = LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24)
    checkpoint = tmp_path / "model.ckpt"
    sample = {"lidar": {"elevation": torch.tensor([-0.2, 0.2]), "azimuth": torch.linspace(-1.0, 1.0, 8)}, "target": {"elevation": torch.tensor([-0.2, 0.0, 0.2])}}
    save_checkpoint(checkpoint, model, epoch=0, global_step=0, sample=sample)
    output = tmp_path / "output"
    command = [sys.executable, "scripts/evaluate_sr.py", "--checkpoint", str(checkpoint), "--dataset-root", str(tmp_path), "--split-file", str(split), "--output-root", str(output), "--device", "cpu"]
    subprocess.run(command, check=True, cwd=".")
    required = {
        "beam_metrics.csv": {"target_row", "range_mae", "operator_entropy", "empty_group"},
        "distance_metrics.csv": {"distance_min_m", "distance_max_m", "region", "range_mae", "operator_entropy", "empty_group"},
        "operator_metrics.csv": {"region", "anchor_mae", "all_invalid_positive_query_ratio", "all_invalid_all_query_ratio", "empty_group"},
    }
    for filename, fields in required.items():
        text = (output / filename).read_text()
        assert "See region" + "_metrics.csv" not in text
        rows = list(csv.DictReader(text.splitlines()))
        assert rows and fields <= set(rows[0])
        assert any(row["empty_group"] == "False" for row in rows)
    assert any(row["range_mae"] != "" for row in csv.DictReader((output / "beam_metrics.csv").open()))

    # Independent raw-tensor recomputation for one beam, one distance bin, and
    # the full-region operator row.  These calculations do not use an evaluator
    # accumulator or values read from the other CSVs.
    restored = LidarOperatorStudent(**torch.load(checkpoint, weights_only=False)["model_config"])
    restored.load_state_dict(torch.load(checkpoint, weights_only=False)["model"])
    restored.eval()
    batch = collate_frames([ProcessedTrainingDataset(tmp_path, split_file=split, depth_mode="none")[0]])
    with torch.no_grad():
        prediction = restored(batch)
    target_range, target_valid = batch["target"]["range"], batch["target"]["valid"].bool()
    beam_rows = list(csv.DictReader((output / "beam_metrics.csv").open()))
    beam = next(row for row in beam_rows if row["target_row"] == "0")
    beam_error = (prediction.predicted_range[:, :, 0] - target_range[:, :, 0]).abs()
    assert math.isclose(float(beam["range_mae"]), float(beam_error.mean()), rel_tol=1e-6, abs_tol=1e-6)

    generated = build_generated_row_mask(batch["lidar"]["elevation"][0], batch["target"]["elevation"][0]).bool().expand(1, 1, -1, 8)
    distance_rows = list(csv.DictReader((output / "distance_metrics.csv").open()))
    distance = next(row for row in distance_rows if row["distance_min_m"] == "10.0" and row["distance_max_m"] == "20.0" and row["region"] == "full")
    distance_mask = generated & target_valid & target_range.ge(10.0) & target_range.lt(20.0)
    distance_error = (prediction.predicted_range - target_range).abs()[distance_mask]
    assert math.isclose(float(distance["range_mae"]), float(distance_error.mean()), rel_tol=1e-6, abs_tol=1e-6)

    operator_rows = list(csv.DictReader((output / "operator_metrics.csv").open()))
    operator = next(row for row in operator_rows if row["region"] == "full")
    selected = generated.squeeze(1) & target_valid.squeeze(1) & prediction.candidate_valid.any(dim=-1)
    anchor_error = (prediction.anchor_range.squeeze(1) - target_range.squeeze(1)).abs()[selected]
    assert int(operator["operator_query_count"]) == int(selected.sum())
    assert math.isclose(float(operator["anchor_mae"]), float(anchor_error.mean()), rel_tol=1e-6, abs_tol=1e-6)
