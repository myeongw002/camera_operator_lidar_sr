"""Schema-3 checkpoints containing NumPy RNG state load through every CLI path."""
from resume_test_support import checkpoint, common, make_dataset, rewrite_checkpoint, run


def test_schema3_checkpoint_loads_in_all_internal_cli_paths(tmp_path):
    make_dataset(tmp_path)
    run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "1"])
    student = checkpoint(tmp_path, "student")
    run(["scripts/train_teacher.py", "--baseline-checkpoint", str(student), "--depth-mode", "correct", *common(tmp_path, "teacher"), "--epochs", "1"])
    teacher = checkpoint(tmp_path, "teacher")
    run(["scripts/evaluate_sr.py", "--checkpoint", str(student), "--dataset-root", str(tmp_path), "--split-file", str(tmp_path / "val.txt"), "--output-root", str(tmp_path / "eval"), "--device", "cpu"])
    run(["scripts/infer.py", "--checkpoint", str(student), "--frame-root", str(tmp_path / "00" / "000000"), "--output", str(tmp_path / "infer.npz"), "--device", "cpu"])
    run(["scripts/train_distill.py", "--baseline", str(student), "--teacher", str(teacher), "--depth-mode", "correct", *common(tmp_path, "distill"), "--epochs", "1"])
    run(["scripts/evaluate_teacher.py", "--baseline-checkpoint", str(student), "--teacher-correct", str(teacher), "--dataset-root", str(tmp_path), "--split-file", str(tmp_path / "val.txt"), "--output-root", str(tmp_path / "teacher_eval"), "--device", "cpu"])


def test_schema2_resume_is_rejected_before_training(tmp_path):
    make_dataset(tmp_path)
    run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "1"])
    rewrite_checkpoint(checkpoint(tmp_path, "student"), lambda value: value.update(checkpoint_schema_version=2))
    result = run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "2", "--resume"], check=False)
    assert result.returncode != 0 and "schema version 3" in result.stderr
