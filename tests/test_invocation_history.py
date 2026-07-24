import hashlib
import json

from resume_test_support import checkpoint, common, json_lines, load, make_dataset, run


def test_invocation_history_and_immutable_experiment_config_for_all_stages(tmp_path):
    make_dataset(tmp_path); run(["scripts/train_student.py", *common(tmp_path, "student"), "--epochs", "1"])
    baseline = checkpoint(tmp_path, "student"); run(["scripts/train_teacher.py", "--baseline-checkpoint", str(baseline), "--depth-mode", "correct", *common(tmp_path, "teacher"), "--epochs", "1"])
    teacher = checkpoint(tmp_path, "teacher"); stage_commands = {
        "student": ["scripts/train_student.py"],
        "teacher": ["scripts/train_teacher.py", "--baseline-checkpoint", str(baseline), "--depth-mode", "correct"],
        "distill": ["scripts/train_distill.py", "--baseline", str(baseline), "--teacher", str(teacher), "--depth-mode", "correct"],
    }
    run([*stage_commands["distill"], *common(tmp_path, "distill"), "--epochs", "1"])
    # Resume consumers before the source student changes its own last checkpoint identity.
    for name in ("distill", "teacher", "student"):
        command = stage_commands[name]
        root = tmp_path / "outputs" / name / "seed_17"; before = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
        run([*command, *common(tmp_path, name), "--epochs", "2", "--resume"])
        assert hashlib.sha256((root / "config.json").read_bytes()).hexdigest() == before
        rows = json_lines(root / "invocations.jsonl"); current = load(root / "checkpoints" / "last.ckpt")
        assert [row["invocation_index"] for row in rows] == [0, 1]
        assert rows[0]["resume"] is False and rows[1]["resume"] is True
        assert current["experiment_config"] == json.loads((root / "config.json").read_text())
        assert current["current_invocation"] == rows[-1]
