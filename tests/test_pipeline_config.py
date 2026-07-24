import pytest

from camera_operator_sr.pipeline.config import load_config, validate_config
from pipeline_support import synthetic_config, write_config


def test_valid_config_and_strict_unknown_key(tmp_path):
    assert load_config(write_config(tmp_path))["pipeline"]["name"] == "synthetic"
    config = synthetic_config(tmp_path); config["student"]["learnig_rate"] = 1
    with pytest.raises(ValueError, match="Did you mean"): validate_config(config)


def test_config_rejects_invalid_modes_rows_overlap_and_name(tmp_path):
    config = synthetic_config(tmp_path); config["teachers"]["correct"]["depth_mode"] = "bad"
    with pytest.raises(ValueError, match="depth mode"): validate_config(config)
    config = synthetic_config(tmp_path); config["splits"]["allow_sequence_overlap"] = False
    with pytest.raises(ValueError, match="sequence overlap"): validate_config(config)
    config = synthetic_config(tmp_path); config["dataset"]["input_row_indices"] = [0, 0]
    with pytest.raises(ValueError, match="unique"): validate_config(config)
