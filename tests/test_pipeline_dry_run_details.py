from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import write_config


def test_dry_run_prints_commands_and_checkpoint_wiring_without_writes(tmp_path, capsys):
    path = write_config(tmp_path); assert PipelineRunner(__import__("camera_operator_sr.pipeline.config", fromlist=["load_config"]).load_config(path), path, dry_run=True).run() == 0
    output = capsys.readouterr().out
    assert "command:" in output and "baseline checkpoint:" in output and "teacher checkpoint:" in output
    assert not (tmp_path / "outputs" / "synthetic").exists()
