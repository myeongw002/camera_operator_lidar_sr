import json
from camera_operator_sr.pipeline.state import PipelineState


def test_state_is_atomic_and_records_stage_status(tmp_path):
    state = PipelineState(tmp_path/"state.json", "test", resume=False); state.update("P00_preflight", "RUNNING"); state.update("P00_preflight", "SUCCEEDED", return_code=0); state.finish("SUCCEEDED")
    loaded = json.loads((tmp_path/"state.json").read_text()); assert loaded["status"] == "SUCCEEDED" and loaded["stages"]["P00_preflight"]["status"] == "SUCCEEDED"
