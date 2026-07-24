from copy import deepcopy
from pathlib import Path

import yaml


def synthetic_config(tmp_path: Path) -> dict:
    source = yaml.safe_load(Path("configs/pipeline/synthetic_test.yaml").read_text())
    source["pipeline"]["output_root"] = str(tmp_path / "outputs")
    source["pipeline"]["name"] = "synthetic"
    source["dataset"]["processed_root"] = str(tmp_path / "processed")
    source["splits"]["root"] = str(tmp_path / "outputs" / "synthetic" / "splits")
    return source


def write_config(tmp_path: Path, config: dict | None = None) -> Path:
    config = synthetic_config(tmp_path) if config is None else config
    path = tmp_path / "pipeline.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False)); return path
