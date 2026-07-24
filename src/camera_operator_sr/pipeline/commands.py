"""Command builders matching the repository's public script CLIs."""
from __future__ import annotations

import sys
from pathlib import Path


def python(script: str) -> list[str]: return [sys.executable, script]
def training_common(context, section: dict, mode: str | None = None) -> list[str]:
    return ["--dataset-root", str(context.processed_root), "--train-split", str(context.train_split), "--val-split", str(context.validation_split), "--output-root", str(context.experiments_root), "--experiment-name", section["experiment_name"], "--seed", str(context.config["pipeline"]["seed"]), "--epochs", str(section.get("epochs", 1)), "--device", context.device] + (["--deterministic"] if context.config["pipeline"].get("deterministic") else []) + ([f"--{mode}"] if mode else [])

def prepare(context, sequence: str) -> list[str]:
    cfg, data = context.config, context.config["dataset"]
    calibration_root = context.calibration_file(sequence).parent
    return python("scripts/prepare_range_images.py") + ["--scan-root", str(context.scan_directory(sequence)), "--output-root", str(context.processed_root/sequence), "--target-elevation", str(context.target_elevation_file), "--input-row-indices", ",".join(map(str, data["input_row_indices"])), "--width", str(data.get("range_width", 2048)), "--calibration-root", str(calibration_root), "--image-root", str(context.image_directory(sequence)), "--camera-id", str(data.get("camera_id", 2)), "--frame-list", str(context.frame_manifest(sequence))]

def depth(context, sequence: str, *, force: bool = False) -> list[str]:
    cfg, data = context.config["depth"], context.config["dataset"]
    device = 0 if str(cfg.get("device", context.device)).startswith("cuda") else -1
    return python("scripts/precompute_relative_depth.py") + ["--image-root", str(context.image_directory(sequence)), "--output-root", str(context.processed_root/sequence), "--model", cfg["model"], "--device", str(device), "--frame-list", str(context.frame_manifest(sequence)), "--batch-size", str(cfg["batch_size"])] + (["--overwrite"] if cfg.get("overwrite") or force else [])

def student(context, mode: str | None = None) -> list[str]:
    cfg = context.config["student"]
    return python("scripts/train_student.py") + training_common(context, cfg, mode) + ["--batch-size", str(cfg.get("batch_size", 2)), "--learning-rate", str(cfg.get("learning_rate", 3e-4)), "--weight-decay", str(cfg["weight_decay"])]

def teacher(context, name: str, mode: str | None = None) -> list[str]:
    cfg = context.config["teachers"][name]
    return python("scripts/train_teacher.py") + ["--baseline-checkpoint", str(context.artifacts["student_best_checkpoint"]), "--depth-mode", cfg["depth_mode"]] + training_common(context, cfg, mode) + ["--batch-size", str(cfg.get("batch_size", 1)), "--learning-rate", str(cfg.get("learning_rate", 2e-4))]

def distillation(context, mode: str | None = None) -> list[str]:
    cfg = context.config["distillation"]
    return python("scripts/train_distill.py") + ["--baseline", str(context.artifacts["student_best_checkpoint"]), "--teacher", str(context.artifacts["teacher_checkpoints"][cfg["teacher"]]), "--depth-mode", cfg["depth_mode"], "--advantage-mode", cfg.get("advantage_mode", "soft")] + training_common(context, cfg, mode) + ["--batch-size", str(cfg.get("batch_size", 1)), "--learning-rate", str(cfg.get("learning_rate", 1e-4))]

def teacher_evaluation(context) -> list[str]:
    command = python("scripts/evaluate_teacher.py") + ["--baseline-checkpoint", str(context.artifacts["student_best_checkpoint"]), "--teacher-correct", str(context.artifacts["teacher_checkpoints"]["correct"]), "--dataset-root", str(context.processed_root), "--split-file", str(context.test_split), "--output-root", str(context.evaluations_root/"teachers"), "--device", context.device]
    for name, flag in (("none", "--teacher-none"), ("frame_shuffled", "--teacher-frame-shuffled"), ("spatial_shuffled", "--teacher-spatial-shuffled"), ("constant", "--teacher-constant"), ("oracle", "--teacher-oracle")):
        if name in context.artifacts.get("teacher_checkpoints", {}): command += [flag, str(context.artifacts["teacher_checkpoints"][name])]
    return command

def sr_evaluation(context, checkpoint: Path, name: str) -> list[str]:
    bins = context.config["evaluation"].get("distance_bins", [])
    return python("scripts/evaluate_sr.py") + ["--checkpoint", str(checkpoint), "--dataset-root", str(context.processed_root), "--split-file", str(context.test_split), "--output-root", str(context.evaluations_root/name), "--device", context.device] + (["--distance-bins", *map(str, bins)] if bins else [])

def inference(context, checkpoint: Path, frame: Path, output: Path) -> list[str]:
    return python("scripts/infer.py") + ["--checkpoint", str(checkpoint), "--frame-root", str(frame), "--output", str(output), "--device", context.device]
