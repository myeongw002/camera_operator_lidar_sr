import csv
import json

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
    assert summary["metadata"]["available_metrics"] == ["range_mae", "range_rmse", "range_count"]
    assert summary["metadata"]["unavailable_metrics"] == ["return", "residual", "operator"]
    beam_rows = list(csv.DictReader((output / "beam_metrics.csv").open()))
    assert set(beam_rows[0]) == {
        "target_row", "target_elevation_deg", "is_observed_row", "is_generated_row",
        "range_mae", "range_rmse", "range_count",
    }
    observed = [row for row in beam_rows if row["is_observed_row"] == "True"]
    generated = [row for row in beam_rows if row["is_generated_row"] == "True"]
    assert all(row["range_count"] == "0" and row["range_mae"] == "" for row in observed)
    assert len(generated) == 1 and generated[0]["range_count"] == "8"
