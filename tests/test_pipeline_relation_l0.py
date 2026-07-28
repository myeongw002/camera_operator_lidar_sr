from pathlib import Path

from camera_operator_sr.pipeline import commands
from camera_operator_sr.pipeline.config import load_config, validate_config
from camera_operator_sr.pipeline.runner import PipelineContext, PipelineRunner
from pipeline_support import synthetic_config, write_config


def configure_l0_only(config):
    config["student"].update(model_type="relation_l0", experiment_name="relation_l0")
    for teacher in config["teachers"].values():
        teacher["enabled"] = False
    config["distillation"]["enabled"] = False
    config["evaluation"].update(model_type="relation_l0", evaluate_teachers=False, evaluate_student=True, evaluate_distilled=False)
    config["inference"]["checkpoint"] = "student"


def test_relation_l0_pipeline_dry_run_uses_only_l0_commands_and_artifacts(tmp_path):
    config = synthetic_config(tmp_path)
    configure_l0_only(config)
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
    configure_l0_only(config)
    config["inference"]["max_frames"] = 1
    config["stages"].update(precompute_depth=False, train_teachers=False, evaluate_teachers=False, train_distillation=False)
    config = validate_config(config)
    path = write_config(tmp_path, config)
    runner = PipelineRunner(config, path)
    assert runner.run() == 0
    root = runner.context.root
    assert (root / "experiments" / "relation_l0" / "seed_17" / "checkpoints" / "last.ckpt").exists()
    assert (root / "evaluations" / "student_baseline" / "relation_metrics.csv").exists()
    assert list((root / "inference" / "student" / "00").glob("*.npz"))


def test_relation_l0_config_rejects_non_l0_only_combinations_and_accepts_pilot(tmp_path):
    assert load_config("configs/pipeline/kitti_relation_l0_pilot.yaml")["student"]["model_type"] == "relation_l0"
    changes = (
        ("teacher", lambda config: config["teachers"]["correct"].update(enabled=True), "teachers must be disabled"),
        ("distillation", lambda config: config["distillation"].update(enabled=True), "distillation is not implemented"),
        ("teacher evaluation", lambda config: config["evaluation"].update(evaluate_teachers=True), "evaluate_teachers"),
        ("distilled evaluation", lambda config: config["evaluation"].update(evaluate_distilled=True), "distillation is not implemented"),
        ("inference", lambda config: config["inference"].update(checkpoint="distillation"), "inference.checkpoint=student"),
    )
    for _, change, message in changes:
        config = synthetic_config(tmp_path)
        configure_l0_only(config)
        change(config)
        try:
            validate_config(config)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid relation_l0 config was accepted")


def test_relation_guided_pipeline_routes_p06_and_p08_without_controls_or_kd(tmp_path):
    config = synthetic_config(tmp_path)
    configure_l0_only(config)
    config["teachers"]["correct"].update(enabled=True, model_type="relation_guided", experiment_name="relation_guided", depth_mode="correct")
    config["evaluation"].update(evaluate_teachers=True)
    config["stages"].update(precompute_depth=False, train_teachers=True, evaluate_teachers=True, train_distillation=False)
    config = validate_config(config)
    path = write_config(tmp_path, config)
    runner = PipelineRunner(config, path, dry_run=True)
    runner._planned_artifacts()
    assert any(value.endswith("train_relation_guided.py") for value in commands.teacher(runner.context, "correct"))
    assert any(value.endswith("evaluate_relation_guided.py") for value in commands.teacher_evaluation(runner.context))
    assert "P07_train_teacher_controls" in runner.disabled
    assert "P09_train_distillation" in runner.disabled


def test_relation_guided_synthetic_pipeline_trains_and_evaluates_g(tmp_path):
    config = synthetic_config(tmp_path)
    configure_l0_only(config)
    config["inference"]["max_frames"] = 1
    config["teachers"]["correct"].update(enabled=True, model_type="relation_guided", experiment_name="relation_guided", depth_mode="correct")
    config["evaluation"].update(evaluate_teachers=True)
    config["stages"].update(precompute_depth=False, train_teachers=True, evaluate_teachers=True, train_distillation=False)
    config = validate_config(config)
    path = write_config(tmp_path, config)
    assert PipelineRunner(config, path).run() == 0
    root = Path(config["pipeline"]["output_root"]) / config["pipeline"]["name"]
    assert (root / "experiments" / "relation_guided" / "seed_17" / "checkpoints" / "last.ckpt").exists()
    assert (root / "evaluations" / "teachers" / "guided_relation_metrics.csv").exists()
