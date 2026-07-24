import sys

from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import synthetic_config, write_config


def test_subprocess_stdout_and_stderr_are_teed_to_terminal_and_logs(tmp_path, capsys):
    config = synthetic_config(tmp_path); path = write_config(tmp_path, config)
    runner = PipelineRunner(config, path)
    command = [sys.executable, "-c", "import sys; print('stage standard output'); print('stage standard error', file=sys.stderr)"]
    assert runner._run_command("P99_test", command) == 0
    captured = capsys.readouterr()
    assert "stage standard output" in captured.out and "stage standard error" in captured.err
    logs = runner.context.root / "logs"
    assert "stage standard output" in (logs / "P99_test.stdout.log").read_text()
    assert "stage standard error" in (logs / "P99_test.stderr.log").read_text()
