from camera_operator_sr.pipeline import commands
from camera_operator_sr.pipeline.config import validate_config
from camera_operator_sr.pipeline.runner import PipelineContext, PipelineRunner
from pipeline_support import synthetic_config, write_config


def test_relation_l0_pipeline_dry_run_uses_only_l0_commands_and_artifacts(tmp_path):
    config = synthetic_config(tmp_path)
    config["student"].update(model_type="relation_l0", experiment_name="relation_l0")
    config["teachers"]["correct"]["enabled"] = False
    config["distillation"]["enabled"] = False
    config["evaluation"].update(model_type="relation_l0", evaluate_teachers=False, evaluate_student=True, evaluate_distilled=False)
    config["inference"]["checkpoint"] = "student"
    config["stages"].update(precompute_depth=False, train_teachers=False, evaluate_teachers=False, train_distillation=False)
    config = validate_config(config)
    path = write_config(tmp_path, config)
    runner = PipelineRunner(config, path, dry_run=True)
    runner._planned_artifacts()
    context = runner.context
    assert any(value.endswith("train_relation_l0.py") for value in commands.student(context))
    assert any(value.endswith("evaluate_relation.py") for value in commands.sr_evaluation(context, context.artifacts["student_best_checkpoint"], "student"))
    assert all("train_teacher.py" not in " ".join(command) and "train_distill.py" not in " ".join(command) for stage in ("P05_train_student", "P10_evaluate_sr", "P11_inference") for command in runner._dry_commands(stage))
    expected = runner._expected("P10_evaluate_sr")
    assert any(path.name == "relation_metrics.csv" for path in expected)


def test_relation_l0_synthetic_pipeline_runs_train_evaluate_and_infer(tmp_path):
    config = synthetic_config(tmp_path)
    config["student"].update(model_type="relation_l0", experiment_name="relation_l0")
    config["teachers"]["correct"]["enabled"] = False
    config["distillation"]["enabled"] = False
    config["evaluation"].update(model_type="relation_l0", evaluate_teachers=False, evaluate_student=True, evaluate_distilled=False)
    config["inference"].update(checkpoint="student", max_frames=1)
    config["stages"].update(precompute_depth=False, train_teachers=False, evaluate_teachers=False, train_distillation=False)
    config = validate_config(config)
    path = write_config(tmp_path, config)
    runner = PipelineRunner(config, path)
    assert runner.run() == 0
    root = runner.context.root
    assert (root / "experiments" / "relation_l0" / "seed_17" / "checkpoints" / "last.ckpt").exists()
    assert (root / "evaluations" / "student_baseline" / "relation_metrics.csv").exists()
    assert list((root / "inference" / "student" / "00").glob("*.npz"))
