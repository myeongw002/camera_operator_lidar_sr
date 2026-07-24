from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import synthetic_config, write_config


def test_enable_flags_disable_corresponding_stages(tmp_path):
    cfg = synthetic_config(tmp_path); cfg["evaluation"].update(enabled=False); cfg["inference"]["enabled"] = False; cfg["teachers"]["none"]["enabled"] = False
    path = write_config(tmp_path, cfg); runner = PipelineRunner(cfg, path)
    assert {"P08_evaluate_teachers", "P10_evaluate_sr", "P11_inference", "P07_train_teacher_controls"} <= runner.disabled
