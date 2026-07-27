import json

from camera_operator_sr.pipeline.config import load_config
from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import synthetic_config, write_config


def test_failed_stage_blocks_downstream_writes_summary_and_reports_terminal_error(tmp_path, monkeypatch, capsys):
    config = synthetic_config(tmp_path); path = write_config(tmp_path, config); runner = PipelineRunner(load_config(path), path)
    original = runner._execute_stage
    def fail(stage):
        if stage == "P05_train_student": raise RuntimeError("intentional failure")
        return original(stage)
    monkeypatch.setattr(runner, "_execute_stage", fail)
    assert runner.run() == 1
    state = json.loads((runner.context.root/"pipeline_state.json").read_text())
    assert state["status"] == "FAILED" and state["stages"]["P05_train_student"]["status"] == "FAILED"
    assert state["stages"]["P06_train_teacher_correct"]["status"] == "BLOCKED"
    assert (runner.context.summary_root/"pipeline_summary.json").exists()
    terminal_error = capsys.readouterr().err
    assert "P05_train_student: FAILED: intentional failure" in terminal_error and "Traceback" in terminal_error
    assert "intentional failure" in (runner.context.root / "logs" / "P05_train_student.stderr.log").read_text()
