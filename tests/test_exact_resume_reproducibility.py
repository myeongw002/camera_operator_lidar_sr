from resume_test_support import checkpoint, common, equal_values, json_lines, load, make_dataset, run


def test_student_exact_resume_matches_continuous_training(tmp_path):
    make_dataset(tmp_path)
    run(["scripts/train_student.py", *common(tmp_path, "continuous"), "--epochs", "2"])
    run(["scripts/train_student.py", *common(tmp_path, "resumed"), "--epochs", "1"])
    run(["scripts/train_student.py", *common(tmp_path, "resumed"), "--epochs", "2", "--resume"])
    continuous, resumed = load(checkpoint(tmp_path, "continuous")), load(checkpoint(tmp_path, "resumed"))
    assert equal_values(continuous["model"], resumed["model"])
    assert equal_values(continuous["optimizer"], resumed["optimizer"])
    for key in ("epoch", "global_step", "current_validation_score", "best_validation_score", "best_epoch", "best_global_step"):
        assert continuous[key] == resumed[key]
    assert json_lines(tmp_path / "outputs" / "continuous" / "seed_17" / "metrics.jsonl") == json_lines(tmp_path / "outputs" / "resumed" / "seed_17" / "metrics.jsonl")
