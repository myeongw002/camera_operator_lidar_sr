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
