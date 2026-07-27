"""Strict pipeline configuration loading and inexpensive validation."""
from __future__ import annotations

import json
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path

import yaml


ALLOWED = {
    "": {"schema_version", "pipeline", "dataset", "depth", "splits", "student", "teachers", "distillation", "evaluation", "inference", "stages"},
    "pipeline": {"name", "output_root", "seed", "device", "deterministic", "fail_fast", "resume", "overwrite", "allow_device_fallback"},
    "dataset": {"type", "raw_root", "scan_root", "image_root", "calibration_root", "processed_root", "sequences", "camera_id", "range_width", "target_elevation_file", "auto_estimate_elevation", "input_row_indices", "max_frames_per_sequence"},
    "depth": {"enabled", "model", "device", "batch_size", "overwrite"},
    "splits": {"root", "train_file", "validation_file", "test_file", "generate", "allow_sequence_overlap"},
    "student": {"enabled", "model_type", "experiment_name", "epochs", "batch_size", "learning_rate", "weight_decay", "horizontal_radius", "point_hidden_dim", "relation_hidden_dim", "correction_limit", "correction_reg_weight", "use_intensity"},
    "distillation": {"enabled", "experiment_name", "teacher", "depth_mode", "advantage_mode", "epochs", "batch_size", "learning_rate"},
    "evaluation": {"enabled", "model_type", "evaluate_teachers", "evaluate_student", "evaluate_distilled", "distance_bins"},
    "inference": {"enabled", "max_frames", "checkpoint"},
    "stages": {"prepare_range_images", "precompute_depth", "create_splits", "train_student", "train_teachers", "evaluate_teachers", "train_distillation", "evaluate_sr", "inference"},
}
TEACHER_ALLOWED = {"enabled", "experiment_name", "depth_mode", "epochs", "batch_size", "learning_rate"}
DEPTH_MODES = {"correct", "none", "frame_shuffled", "spatial_shuffled", "constant", "oracle"}


def _unknown(section: str, value: dict, allowed: set[str]) -> None:
    for key in value:
        if key not in allowed:
            suggestion = get_close_matches(key, allowed, n=1)
            message = f"Unknown pipeline config key: {section + '.' if section else ''}{key}"
            if suggestion:
                message += f"\nDid you mean: {section + '.' if section else ''}{suggestion[0]}"
            raise ValueError(message)


def _mapping(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Pipeline config requires mapping: {key}")
    return value


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict): raise ValueError("Pipeline config must be a mapping")
    _unknown("", config, ALLOWED[""])
    if config.get("schema_version") != 1: raise ValueError("pipeline schema_version must be 1")
    for section in ("pipeline", "dataset", "depth", "splits", "student", "distillation", "evaluation", "inference", "stages"):
        _unknown(section, _mapping(config, section), ALLOWED[section])
    _mapping(config, "teachers")
    pipeline, dataset, splits = config["pipeline"], config["dataset"], config["splits"]
    for section, keys in ((pipeline, {"name", "output_root", "seed", "device"}), (dataset, {"type", "processed_root", "sequences"}), (splits, {"root", "train_file", "validation_file", "test_file"})):
        missing = keys - section.keys()
        if missing: raise ValueError("Pipeline config missing required key: " + sorted(missing)[0])
    if not pipeline["name"] or Path(str(pipeline["name"])).name != pipeline["name"]: raise ValueError("invalid pipeline name")
    sequences = dataset["sequences"]
    if not isinstance(sequences, dict): raise ValueError("dataset.sequences must be a mapping")
    sequence_sets = {name: set(sequences.get(name, [])) for name in ("train", "validation", "test")}
    if not splits.get("allow_sequence_overlap", False):
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if sequence_sets[left] & sequence_sets[right]: raise ValueError(f"sequence overlap: {left}/{right}")
    indices = dataset.get("input_row_indices", [])
    if indices and (len(indices) != len(set(indices)) or any(not isinstance(value, int) or value < 0 or value >= 64 for value in indices)):
        raise ValueError("dataset.input_row_indices must contain unique indices in [0, 64)")
    for name, teacher in config["teachers"].items():
        if not isinstance(teacher, dict): raise ValueError(f"teachers.{name} must be a mapping")
        _unknown(f"teachers.{name}", teacher, TEACHER_ALLOWED)
        if teacher.get("depth_mode") not in DEPTH_MODES: raise ValueError(f"invalid teacher depth mode: {teacher.get('depth_mode')}")
    if config["distillation"].get("depth_mode") not in DEPTH_MODES: raise ValueError("invalid distillation depth mode")
    if config["distillation"].get("teacher") not in config["teachers"]: raise ValueError("distillation.teacher must name a configured teacher")
    if config["depth"].get("batch_size", 1) != 1:
        raise ValueError("depth.batch_size must be 1: the selected depth backend is single-image")
    if config["student"].get("weight_decay", 0.0) < 0:
        raise ValueError("student.weight_decay must be non-negative")
    if config["student"].get("model_type", "legacy_operator") not in {"legacy_operator", "relation_l0"}:
        raise ValueError("student.model_type must be legacy_operator or relation_l0")
    if config["evaluation"].get("model_type", config["student"].get("model_type", "legacy_operator")) not in {"legacy_operator", "relation_l0"}:
        raise ValueError("evaluation.model_type must be legacy_operator or relation_l0")
    if config["student"].get("model_type") == "relation_l0" and config["evaluation"].get("model_type", "relation_l0") != "relation_l0":
        raise ValueError("relation_l0 student requires relation_l0 evaluation")
    selected_teacher = config["distillation"].get("teacher")
    if config["distillation"].get("enabled", True) and not config["teachers"][selected_teacher].get("enabled", False):
        raise ValueError("distillation.enabled requires its selected teacher to be enabled")
    evaluation = config["evaluation"]
    if evaluation.get("evaluate_distilled", False) and not config["distillation"].get("enabled", True):
        raise ValueError("evaluation.evaluate_distilled requires distillation.enabled")
    if evaluation.get("evaluate_teachers", False) and not config["teachers"].get("correct", {}).get("enabled", False):
        raise ValueError("evaluation.evaluate_teachers requires teachers.correct.enabled")
    if dataset.get("type") not in {"synthetic", "processed_synthetic"}:
        split_roots = (dataset.get("scan_root"), dataset.get("image_root"), dataset.get("calibration_root"))
        if not dataset.get("raw_root") and not all(split_roots):
            raise ValueError("KITTI dataset requires raw_root or scan_root, image_root, and calibration_root")
    return deepcopy(config)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    text = path.read_text()
    raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return validate_config(raw)
