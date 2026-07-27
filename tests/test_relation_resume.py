from resume_test_support import checkpoint, common, load, make_dataset, run


def test_relation_l0_train_resume_preserves_schema_optimizer_and_progress(tmp_path):
    make_dataset(tmp_path)
    args = ["scripts/train_relation_l0.py", *common(tmp_path, "relation")]
    run([*args, "--epochs", "1"])
    first = load(checkpoint(tmp_path, "relation"))
    run([*args, "--epochs", "2", "--resume"])
    second = load(checkpoint(tmp_path, "relation"))
    assert first["checkpoint_schema_version"] == second["checkpoint_schema_version"] == 4
    assert second["model_config"]["model_type"] == "relation_l0"
    assert second["epoch"] > first["epoch"] and second["global_step"] > first["global_step"]
    assert second["optimizer"] and second["rng_state"] is not None
    evaluation = tmp_path / "evaluation"
    run(["scripts/evaluate_relation.py", "--checkpoint", str(checkpoint(tmp_path, "relation")), "--dataset-root", str(tmp_path), "--split-file", str(tmp_path / "val.txt"), "--output-root", str(evaluation), "--device", "cpu", "--distance-bins", "0", "20", "inf"])
    inference = tmp_path / "inference.npz"
    run(["scripts/infer.py", "--checkpoint", str(checkpoint(tmp_path, "relation")), "--frame-root", str(tmp_path / "00" / "000000"), "--output", str(inference), "--device", "cpu"])
    assert (evaluation / "summary.json").exists() and inference.exists()
