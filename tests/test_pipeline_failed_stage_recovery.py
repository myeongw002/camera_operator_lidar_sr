import json
import os
import subprocess
import sys

from pipeline_support import write_config


def test_failed_training_recovers_with_resume_or_quarantine(tmp_path):
    config = write_config(tmp_path); env = dict(os.environ, PYTHONPATH="src")
    assert subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config)], env=env).returncode == 0
    root = tmp_path / "outputs" / "synthetic"; state_path = root / "pipeline_state.json"
    state = json.loads(state_path.read_text()); state["stages"]["P05_train_student"]["status"] = "FAILED"; state_path.write_text(json.dumps(state))
    recovered = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config), "--resume", "--only-stage", "P05_train_student"], env=env, text=True, capture_output=True)
    assert recovered.returncode == 0 and "--resume" in recovered.stdout
    state = json.loads(state_path.read_text()); state["stages"]["P05_train_student"]["status"] = "FAILED"; state_path.write_text(json.dumps(state))
    experiment = root / "experiments" / "student_baseline" / "seed_17"; (experiment / "checkpoints" / "last.ckpt").unlink()
    fresh = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", str(config), "--resume", "--only-stage", "P05_train_student"], env=env, text=True, capture_output=True)
    assert fresh.returncode == 0 and "--overwrite" in fresh.stdout
    assert list((experiment.parent).glob("seed_17.failed-*"))
