import torch

from resume_test_support import checkpoint, common, load, make_dataset, run


def test_guided_training_resume_preserves_frozen_l0_and_advances_progress(tmp_path):
    make_dataset(tmp_path)
    run(["scripts/train_relation_l0.py", *common(tmp_path, "l0"), "--epochs", "1"])
    l0_path = checkpoint(tmp_path, "l0")
    l0_before = {key: value.clone() for key, value in load(l0_path)["model"].items()}
    guided = ["scripts/train_relation_guided.py", *common(tmp_path, "guided"), "--l0-checkpoint", str(l0_path), "--epochs", "1", "--depth-mode", "correct"]
    run(guided)
    run([*guided, "--epochs", "2", "--resume"])
    state = load(checkpoint(tmp_path, "guided"))
    assert state["model_config"]["model_type"] == "relation_guided"
    assert state["validation_metric"] == "camera_guidance_valid_query_weighted_range_mae"
    assert state["epoch"] == 2 and state["global_step"] > 0
    assert state["source_checkpoints"]["l0"]["path"] == str(l0_path)
    assert all(torch.equal(value, load(l0_path)["model"][key]) for key, value in l0_before.items())
