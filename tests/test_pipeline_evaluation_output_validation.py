import json
import pytest

from camera_operator_sr.pipeline.validation import validate_csv_metrics


def test_metric_csv_validation_rejects_missing_required_header(tmp_path):
    path = tmp_path / "metrics.csv"; path.write_text("range_mae\n1.0\n")
    with pytest.raises(ValueError): validate_csv_metrics(path)
    path.write_text("empty_group,range_mae\nFalse,1.0\n"); validate_csv_metrics(path)
