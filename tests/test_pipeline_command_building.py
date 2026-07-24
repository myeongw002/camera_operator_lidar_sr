from camera_operator_sr.pipeline import commands
from camera_operator_sr.pipeline.runner import PipelineContext
from pipeline_support import synthetic_config, write_config
from pathlib import Path


def test_commands_use_actual_cli_and_automatic_checkpoint_wiring(tmp_path):
    config = synthetic_config(tmp_path); context = PipelineContext(config, tmp_path/"run", write_config(tmp_path), "cpu")
    context.artifacts["student_best_checkpoint"] = tmp_path/"student.ckpt"; context.artifacts["teacher_checkpoints"]["correct"] = tmp_path/"teacher.ckpt"
    assert "--baseline-checkpoint" in commands.teacher(context, "correct") and str(tmp_path/"student.ckpt") in commands.teacher(context, "correct")
    distill = commands.distillation(context); assert distill[distill.index("--teacher") + 1] == str(tmp_path/"teacher.ckpt")
    assert "--checkpoint" in commands.sr_evaluation(context, tmp_path/"student.ckpt", "student")
