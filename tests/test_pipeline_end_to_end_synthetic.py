import json
import os
import subprocess
import sys

from pipeline_support import write_config


def test_synthetic_pipeline_executes_real_training_and_evaluation_subprocesses(tmp_path):
    config = write_config(tmp_path); environment = dict(os.environ); environment["PYTHONPATH"] = "src"
    result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config)], cwd=".", env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    root = tmp_path/"outputs"/"synthetic"; state = json.loads((root/"pipeline_state.json").read_text())
    assert all(value["status"] in {"SUCCEEDED", "SKIPPED"} for value in state["stages"].values())
    for path in (root/"summary"/"pipeline_summary.json", root/"summary"/"artifact_index.json", root/"summary"/"metric_comparison.csv"): assert path.exists()
    before = (root/"experiments"/"student_baseline"/"seed_17"/"checkpoints"/"best.ckpt").read_bytes()
    resumed = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config), "--resume"], cwd=".", env=environment, text=True, capture_output=True)
    assert resumed.returncode == 0, resumed.stderr
    assert before == (root/"experiments"/"student_baseline"/"seed_17"/"checkpoints"/"best.ckpt").read_bytes()
    resumed_again = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config), "--resume"], cwd=".", env=environment, text=True, capture_output=True)
    assert resumed_again.returncode == 0, resumed_again.stderr
    (root / "evaluations" / "student_baseline" / "beam_metrics.csv").unlink()
    repaired_evaluation = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config), "--resume"], cwd=".", env=environment, text=True, capture_output=True)
    assert repaired_evaluation.returncode == 0, repaired_evaluation.stderr
    assert (root / "evaluations" / "student_baseline" / "beam_metrics.csv").exists()
    (root / "inference" / "distillation" / "00" / "000000.npz").unlink()
    repaired_inference = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config), "--resume"], cwd=".", env=environment, text=True, capture_output=True)
    assert repaired_inference.returncode == 0, repaired_inference.stderr
    assert (root / "inference" / "distillation" / "00" / "000000.npz").exists()
