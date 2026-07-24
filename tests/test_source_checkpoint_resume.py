import shutil

from resume_test_support import checkpoint, common, make_dataset, rewrite_checkpoint, run


def test_teacher_resume_rejects_changed_baseline_but_accepts_identical_copy(tmp_path):
    make_dataset(tmp_path); run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "1"])
    baseline = checkpoint(tmp_path, "student")
    copy = tmp_path / "baseline_copy.ckpt"; shutil.copyfile(baseline, copy)
    run(["scripts/train_teacher.py", "--baseline-checkpoint", str(copy), "--depth-mode", "correct", *common(tmp_path, "teacher"), "--epochs", "1"])
    run(["scripts/train_teacher.py", "--baseline-checkpoint", str(copy), "--depth-mode", "correct", *common(tmp_path, "teacher"), "--epochs", "2", "--resume"])
    rewrite_checkpoint(copy, lambda value: value.update(global_step=value["global_step"] + 1))
    result = run(["scripts/train_teacher.py", "--baseline-checkpoint", str(copy), "--depth-mode", "correct", *common(tmp_path, "teacher"), "--epochs", "3", "--resume"], check=False)
    assert result.returncode and "Resume source checkpoint mismatch: baseline" in result.stderr


def test_distillation_resume_rejects_changed_baseline_and_teacher(tmp_path):
    make_dataset(tmp_path); run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "1"])
    baseline = checkpoint(tmp_path, "student"); run(["scripts/train_teacher.py", "--baseline-checkpoint", str(baseline), "--depth-mode", "correct", *common(tmp_path, "teacher"), "--epochs", "1"])
    teacher = checkpoint(tmp_path, "teacher"); args = ["--baseline", str(baseline), "--teacher", str(teacher), "--depth-mode", "correct"]
    run(["scripts/train_distill.py", *args, *common(tmp_path, "distill"), "--epochs", "1"])
    rewrite_checkpoint(baseline, lambda value: value.update(global_step=value["global_step"] + 1))
    baseline_failure = run(["scripts/train_distill.py", *args, *common(tmp_path, "distill"), "--epochs", "2", "--resume"], check=False)
    assert baseline_failure.returncode and "Resume source checkpoint mismatch: baseline" in baseline_failure.stderr
    # Restore the original source identity by training a fresh distillation experiment, then mutate only its teacher.
    make_dataset(tmp_path / "other"); run(["scripts/train_student.py", *common(tmp_path / "other", "student"), "--epochs", "1"])
    b2 = checkpoint(tmp_path / "other", "student"); run(["scripts/train_teacher.py", "--baseline-checkpoint", str(b2), "--depth-mode", "correct", *common(tmp_path / "other", "teacher"), "--epochs", "1"])
    t2 = checkpoint(tmp_path / "other", "teacher"); args2 = ["--baseline", str(b2), "--teacher", str(t2), "--depth-mode", "correct"]
    run(["scripts/train_distill.py", *args2, *common(tmp_path / "other", "distill"), "--epochs", "1"])
    rewrite_checkpoint(t2, lambda value: value.update(global_step=value["global_step"] + 1))
    teacher_failure = run(["scripts/train_distill.py", *args2, *common(tmp_path / "other", "distill"), "--epochs", "2", "--resume"], check=False)
    assert teacher_failure.returncode and "Resume source checkpoint mismatch: teacher" in teacher_failure.stderr
