from camera_operator_sr.pipeline.config import load_config
from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import write_config


def test_dry_run_does_not_create_pipeline_output(tmp_path):
    path = write_config(tmp_path); runner = PipelineRunner(load_config(path), path, dry_run=True)
    assert runner.run() == 0
    assert not runner.context.root.exists()
