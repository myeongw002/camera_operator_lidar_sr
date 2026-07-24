from pathlib import Path

from camera_operator_sr.pipeline import commands
from camera_operator_sr.pipeline.runner import PipelineContext
from pipeline_support import synthetic_config, write_config


def test_student_and_teacher_artifacts_are_wired_without_config_checkpoint_paths(tmp_path):
    config = synthetic_config(tmp_path); context = PipelineContext(config, tmp_path/"run", write_config(tmp_path), "cpu")
    student, teacher = Path("/generated/student.ckpt"), Path("/generated/teacher.ckpt")
    context.artifacts["student_best_checkpoint"] = student; context.artifacts["teacher_checkpoints"]["correct"] = teacher
    teacher_command, distill_command = commands.teacher(context, "correct"), commands.distillation(context)
    assert teacher_command[teacher_command.index("--baseline-checkpoint") + 1] == str(student)
    assert distill_command[distill_command.index("--teacher") + 1] == str(teacher)
