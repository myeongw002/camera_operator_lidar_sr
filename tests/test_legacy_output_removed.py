from resume_test_support import run


def test_training_clis_reject_removed_output_argument(tmp_path):
    commands = (
        ["scripts/train_student.py", "--dataset-root", str(tmp_path), "--output-root", str(tmp_path), "--experiment-name", "student", "--seed", "1", "--output", "old.ckpt"],
        ["scripts/train_teacher.py", "--dataset-root", str(tmp_path), "--baseline-checkpoint", "base.ckpt", "--output-root", str(tmp_path), "--experiment-name", "teacher", "--seed", "1", "--output", "old.ckpt"],
        ["scripts/train_distill.py", "--dataset-root", str(tmp_path), "--baseline", "base.ckpt", "--teacher", "teacher.ckpt", "--output-root", str(tmp_path), "--experiment-name", "distill", "--seed", "1", "--output", "old.ckpt"],
    )
    for command in commands:
        result = run(command, check=False)
        assert result.returncode != 0
        assert "unrecognized arguments: --output old.ckpt" in result.stderr and "AttributeError" not in result.stderr
