import json

from camera_operator_sr.pipeline.runner import PipelineContext
from camera_operator_sr.pipeline.summary import build_summary
from pipeline_support import synthetic_config, write_config


def test_summary_includes_scientific_checks_and_kd_status(tmp_path):
    cfg = synthetic_config(tmp_path); context = PipelineContext(cfg, tmp_path / "run", write_config(tmp_path), "cpu")
    context.evaluations_root.joinpath("student_baseline").mkdir(parents=True); context.evaluations_root.joinpath(cfg["distillation"]["experiment_name"]).mkdir(parents=True)
    for name, mae in (("student_baseline", 2.0), (cfg["distillation"]["experiment_name"], 1.0)):
        (context.evaluations_root / name / "summary.json").write_text(json.dumps({"full": {"range_mae": mae}}))
    metrics = context.experiments_root / cfg["distillation"]["experiment_name"] / "seed_17"; metrics.mkdir(parents=True); (metrics / "metrics.jsonl").write_text(json.dumps({"range_kd_active_ratio": .5, "range_kd_eligible_count": 2, "return_kd_active_ratio": 0, "return_kd_eligible_count": 1}) + "\n")
    build_summary(context, {"status": "SUCCEEDED"}); summary = json.loads((context.summary_root / "pipeline_summary.json").read_text())
    assert summary["scientific_checks"]["distillation_better_than_student"] and summary["scientific_checks"]["range_kd_active"] and not summary["scientific_checks"]["return_kd_active"]
