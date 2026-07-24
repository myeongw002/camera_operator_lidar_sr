from camera_operator_sr.pipeline.state import PipelineState


def test_stage_transition_discards_stale_failure_and_skip_metadata(tmp_path):
    state = PipelineState(tmp_path / "state.json", "x", resume=False); state.update("P05", "SKIPPED", reason="old")
    state.update("P05", "RUNNING", command=["run"]); assert "reason" not in state.value["stages"]["P05"]
    state.update("P05", "FAILED", error="bad", return_code=1); state.update("P05", "RUNNING")
    assert "error" not in state.value["stages"]["P05"] and "return_code" not in state.value["stages"]["P05"]
