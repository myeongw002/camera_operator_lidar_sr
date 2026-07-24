import json

from resume_test_support import checkpoint, common, load, make_dataset, run


def test_loss_config_round_trips_identically_for_every_stage(tmp_path):
    make_dataset(tmp_path); run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "1"])
    baseline = checkpoint(tmp_path, "student"); run(["scripts/train_teacher.py", "--baseline-checkpoint", str(baseline), "--depth-mode", "correct", *common(tmp_path, "teacher"), "--epochs", "1"])
    teacher = checkpoint(tmp_path, "teacher"); run(["scripts/train_distill.py", "--baseline", str(baseline), "--teacher", str(teacher), "--depth-mode", "correct", *common(tmp_path, "distill"), "--epochs", "1"])
    for name in ("student", "teacher", "distill"):
        root = tmp_path / "outputs" / name / "seed_17"; config = json.loads((root / "config.json").read_text())["loss_config"]
        manifest = json.loads((root / "manifest.json").read_text())["loss_config"]
        best, last = load(root / "checkpoints" / "best.ckpt")["loss_config"], load(root / "checkpoints" / "last.ckpt")["loss_config"]
        assert config and config == manifest == best == last
