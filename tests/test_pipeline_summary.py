import csv
import json

from camera_operator_sr.pipeline.summary import build_summary
from camera_operator_sr.pipeline.runner import PipelineContext
from pipeline_support import synthetic_config, write_config


def test_summary_writes_required_artifacts_and_metric_comparison(tmp_path):
    config = synthetic_config(tmp_path); context = PipelineContext(config, tmp_path/"run", write_config(tmp_path), "cpu")
    context.train_entries, context.validation_entries, context.test_entries = ["00/0"], ["00/0"], ["00/0"]
    context.artifacts.update(student_best_checkpoint=tmp_path/"student.ckpt", distillation_best_checkpoint=tmp_path/"distill.ckpt")
    for name, mae in (("student_baseline", 2.0), (config["distillation"]["experiment_name"], 1.0)):
        path = context.evaluations_root/name; path.mkdir(parents=True); (path/"summary.json").write_text(json.dumps({"full": {"range_mae": mae}}))
    build_summary(context, {"status": "SUCCEEDED"})
    assert (context.summary_root/"pipeline_summary.json").exists()
    assert list(csv.DictReader((context.summary_root/"metric_comparison.csv").open()))[0]["metric"] == "range_mae"


def test_guided_summary_uses_camera_guidance_group_without_inactive_kd_warnings(tmp_path):
    config = synthetic_config(tmp_path)
    config["student"].update(model_type="relation_l0", experiment_name="relation_l0")
    config["teachers"]["correct"].update(enabled=True, model_type="relation_guided", experiment_name="relation_guided")
    config["teachers"]["none"]["enabled"] = False
    config["distillation"]["enabled"] = False
    context = PipelineContext(config, tmp_path / "run", write_config(tmp_path, config), "cpu")
    context.artifacts.update(student_best_checkpoint=tmp_path / "l0.ckpt", teacher_checkpoints={"correct": tmp_path / "guided.ckpt"})
    teacher_root = context.evaluations_root / "teachers"; teacher_root.mkdir(parents=True)
    teacher_root.joinpath("summary.json").write_text(json.dumps({"full_generated_supported": {}, "camera_guidance_valid": {"l0_range_mae": 2.0, "guided_range_mae": 1.0, "l0_range_rmse": 3.0, "guided_range_rmse": 2.0, "camera_guidance_count": 4}, "groups": {}, "metadata": {}}))
    build_summary(context, {"status": "SUCCEEDED"})
    summary = json.loads((context.summary_root / "pipeline_summary.json").read_text())
    checks = summary["scientific_checks"]
    assert checks["guided_better_than_l0_on_camera_guidance_mae"]
    assert checks["guided_better_than_l0_on_camera_guidance_rmse"]
    assert checks["guided_camera_guidance_count_positive"]
    assert not any("teacher_none" in message or "distillation" in message or "KD" in message for message in summary["warnings"])
    assert summary["artifacts"]["guided_evaluation"].endswith("evaluations/teachers/summary.json")
    assert "relation_guided" in (context.summary_root / "experiment_table.csv").read_text()
    rows = list(csv.DictReader((context.summary_root / "metric_comparison.csv").open()))
    assert any(row["comparison"] == "l0_vs_guided" and row["region"] == "camera_guidance_valid" for row in rows)


def test_guided_summary_emits_only_guided_warnings_for_worse_camera_metric(tmp_path):
    config = synthetic_config(tmp_path)
    config["student"].update(model_type="relation_l0")
    config["teachers"]["correct"].update(enabled=True, model_type="relation_guided")
    config["teachers"]["none"]["enabled"] = False
    config["distillation"]["enabled"] = False
    context = PipelineContext(config, tmp_path / "run", write_config(tmp_path, config), "cpu")
    root = context.evaluations_root / "teachers"; root.mkdir(parents=True)
    root.joinpath("summary.json").write_text(json.dumps({"camera_guidance_valid": {"l0_range_mae": 1.0, "guided_range_mae": 2.0, "l0_range_rmse": 1.0, "guided_range_rmse": 2.0, "camera_guidance_count": 0}}))
    build_summary(context, {"status": "SUCCEEDED"})
    warnings = json.loads((context.summary_root / "pipeline_summary.json").read_text())["warnings"]
    assert any("relation_guided did not outperform frozen L0" in message for message in warnings)
    assert not any("KD" in message or "distillation" in message or "teacher_none" in message for message in warnings)
