import csv
import json

import numpy as np

from resume_test_support import make_dataset, run


def test_b0_evaluator_is_checkpoint_free_and_reports_generated_range_metrics_only(tmp_path):
    make_dataset(tmp_path)
    output = tmp_path / "b0_eval"
    run([
        "scripts/evaluate_b0.py", "--dataset-root", str(tmp_path),
        "--split-file", str(tmp_path / "val.txt"), "--output-root", str(output),
        "--device", "cpu",
    ])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["metadata"]["checkpoint"] is None
    assert summary["metadata"]["available_metrics"] == ["range_mae", "range_rmse", "supported_count", "anchor_coverage", "unsupported_valid_target_count"]
    assert summary["metadata"]["auxiliary_metrics"] == ["zero_filled_range_mae", "zero_filled_range_rmse", "zero_filled_count"]
    assert summary["metadata"]["unavailable_metrics"] == ["return", "residual", "operator"]
    beam_rows = list(csv.DictReader((output / "beam_metrics.csv").open()))
    assert set(beam_rows[0]) == {
        "target_row", "target_elevation_deg", "is_observed_row", "is_generated_row",
        "range_mae", "range_rmse", "supported_count", "anchor_coverage",
        "unsupported_valid_target_count", "zero_filled_range_mae",
        "zero_filled_range_rmse", "zero_filled_count", "valid_target_count",
    }
    observed = [row for row in beam_rows if row["is_observed_row"] == "True"]
    generated = [row for row in beam_rows if row["is_generated_row"] == "True"]
    assert all(row["supported_count"] == "0" and row["range_mae"] == "" for row in observed)
    assert len(generated) == 1 and generated[0]["supported_count"] == "8"
    assert generated[0]["anchor_coverage"] == "1.0"
    assert summary["global"]["anchor_coverage"] == 1.0


def test_b0_evaluator_counts_out_of_fov_valid_generated_targets_as_unsupported(tmp_path):
    make_dataset(tmp_path)
    frame = tmp_path / "00" / "000000"
    meta = np.load(frame / "meta.npz")
    np.savez(
        frame / "meta.npz", input_elevation=meta["input_elevation"],
        target_elevation=np.array([-.2, 0., .6], dtype=np.float32), azimuth=meta["azimuth"],
        K=meta["K"], T_cam_lidar=meta["T_cam_lidar"], image_size=meta["image_size"],
    )
    output = tmp_path / "b0_out_of_fov"
    run([
        "scripts/evaluate_b0.py", "--dataset-root", str(tmp_path),
        "--split-file", str(tmp_path / "val.txt"), "--output-root", str(output),
        "--device", "cpu",
    ])
    summary = json.loads((output / "summary.json").read_text())["global"]
    assert summary["valid_target_count"] == 16
    assert summary["supported_count"] == 8
    assert summary["unsupported_valid_target_count"] == 8
    assert summary["anchor_coverage"] == 0.5
    assert summary["zero_filled_count"] == 16
    rows = list(csv.DictReader((output / "beam_metrics.csv").open()))
    out_of_fov = next(row for row in rows if row["target_row"] == "2")
    assert out_of_fov["supported_count"] == "0"
    assert out_of_fov["unsupported_valid_target_count"] == "8"
    assert out_of_fov["anchor_coverage"] == "0.0"
    assert out_of_fov["range_mae"] == ""
