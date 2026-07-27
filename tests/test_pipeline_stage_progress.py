from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import synthetic_config, write_config


def test_runner_prints_stage_progress_and_final_status(tmp_path, capsys):
    config = synthetic_config(tmp_path); path = write_config(tmp_path, config)
    runner = PipelineRunner(config, path)
    assert runner.run(only_stage="P00_preflight") == 0
    output = capsys.readouterr().out
    assert "[debug:P00]" in output
    assert "[1/1] P00_preflight: STARTED" in output
    assert "[1/1] P00_preflight: SUCCEEDED" in output
    assert "pipeline status: SUCCEEDED" in output
