from camera_operator_sr.pipeline.config import load_config
from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import write_config


def test_resume_skips_matching_completed_stage_and_force_reruns(tmp_path):
    path = write_config(tmp_path); config = load_config(path)
    assert PipelineRunner(config, path).run(only_stage="P00_preflight") == 0
    assert PipelineRunner(config, path, resume=True).run(only_stage="P00_preflight") == 0
    state = PipelineRunner(config, path, resume=True).state.value
    assert state["stages"]["P00_preflight"]["status"] == "SKIPPED"
    assert PipelineRunner(config, path, resume=True, force_stages={"P00_preflight"}).run(only_stage="P00_preflight") == 0
