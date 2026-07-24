import hashlib

from resume_test_support import checkpoint, common, load, make_dataset, rewrite_checkpoint, run
from camera_operator_sr.training.resume import is_historical_best


def test_historical_best_uses_strict_improvement_and_student_does_not_overwrite_it(tmp_path):
    assert not is_historical_best(.10, .10) and is_historical_best(.05, .10)
    make_dataset(tmp_path); run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "1"])
    best = tmp_path / "outputs" / "student" / "seed_17" / "checkpoints" / "best.ckpt"
    before = hashlib.sha256(best.read_bytes()).hexdigest()
    # Range MAE is non-negative, so 0.0 is a deterministic historical best that a later epoch cannot improve.
    rewrite_checkpoint(checkpoint(tmp_path, "student"), lambda value: value.update(best_validation_score=0.0, best_epoch=1, best_global_step=value["global_step"]))
    run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "2", "--resume"])
    after = load(checkpoint(tmp_path, "student"))
    assert after["best_validation_score"] == 0.0
    assert hashlib.sha256(best.read_bytes()).hexdigest() == before


def test_teacher_and_distillation_checkpoints_include_historical_best_fields(tmp_path):
    make_dataset(tmp_path); run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "1"])
    baseline = checkpoint(tmp_path, "student")
    run(["scripts/train_teacher.py", "--baseline-checkpoint", str(baseline), "--depth-mode", "correct", *common(tmp_path, "teacher"), "--epochs", "1"])
    teacher = checkpoint(tmp_path, "teacher")
    run(["scripts/train_distill.py", "--baseline", str(baseline), "--teacher", str(teacher), "--depth-mode", "correct", *common(tmp_path, "distill"), "--epochs", "1"])
    for path in (teacher, checkpoint(tmp_path, "distill")):
        value = load(path)
        assert all(key in value for key in ("current_validation_score", "best_validation_score", "best_epoch", "best_global_step"))
