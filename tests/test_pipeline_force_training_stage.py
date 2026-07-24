import hashlib
import os
import subprocess
import sys

from pipeline_support import write_config


def test_force_student_overwrites_training_and_reruns_downstream(tmp_path):
    config = write_config(tmp_path); env = dict(os.environ, PYTHONPATH="src")
    assert subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config)], env=env).returncode == 0
    checkpoint = tmp_path / "outputs" / "synthetic" / "experiments" / "student_baseline" / "seed_17" / "checkpoints" / "best.ckpt"
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config), "--resume", "--force-stage", "P05_train_student"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--overwrite" in result.stdout and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != before
    import json
    state = json.loads((tmp_path / "outputs" / "synthetic" / "pipeline_state.json").read_text())
    assert state["stages"]["P04_create_splits"]["status"] == "SKIPPED"
    assert all(state["stages"][stage]["status"] == "SUCCEEDED" for stage in ("P05_train_student", "P06_train_teacher_correct", "P09_train_distillation", "P10_evaluate_sr", "P12_summary"))
