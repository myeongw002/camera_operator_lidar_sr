"""Small real-data subprocess helpers shared by resume integration tests."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def make_dataset(root: Path) -> None:
    for index in range(2):
        frame = root / "00" / f"{index:06d}"
        frame.mkdir(parents=True)
        width = 8
        input_range = np.ones((2, width), np.float32) * (10 + index)
        target_range = np.ones((3, width), np.float32) * (10 + index)
        for name, value in (("input_range.npy", input_range), ("input_intensity.npy", np.zeros_like(input_range)),
                            ("input_valid.npy", np.ones_like(input_range)), ("target_range.npy", target_range),
                            ("target_valid.npy", np.ones_like(target_range)), ("relative_depth.npy", np.ones((16, 16), np.float32)),
                            ("depth_valid.npy", np.ones((16, 16), np.float32))):
            np.save(frame / name, value)
        k = np.array([[10, 0, 8], [0, 10, 8], [0, 0, 1]], np.float32)
        t = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], np.float32)
        np.savez(frame / "meta.npz", input_elevation=np.array([-.2, .2], np.float32),
                 target_elevation=np.array([-.2, 0, .2], np.float32), azimuth=np.linspace(-.3, .3, width, dtype=np.float32),
                 K=k, T_cam_lidar=t, image_size=np.array([16, 16]))
    (root / "train.txt").write_text("00/000000\n00/000001\n")
    (root / "val.txt").write_text("00/000000\n")


def run(arguments: list[str], *, check: bool = True):
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, *arguments], cwd=".", text=True, capture_output=True, env=env)
    if check:
        assert result.returncode == 0, result.stderr
    return result


def common(root: Path, name: str, seed: int = 17) -> list[str]:
    return ["--dataset-root", str(root), "--train-split", str(root / "train.txt"), "--val-split", str(root / "val.txt"),
            "--output-root", str(root / "outputs"), "--experiment-name", name, "--seed", str(seed), "--deterministic", "--device", "cpu"]


def checkpoint(root: Path, name: str, seed: int = 17) -> Path:
    return root / "outputs" / name / f"seed_{seed}" / "checkpoints" / "last.ckpt"


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def equal_values(left, right):
    if isinstance(left, torch.Tensor): return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict): return left.keys() == right.keys() and all(equal_values(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)): return type(left) is type(right) and len(left) == len(right) and all(equal_values(a, b) for a, b in zip(left, right))
    return left == right


def rewrite_checkpoint(path: Path, change) -> None:
    value = load(path); change(value); torch.save(value, path)
