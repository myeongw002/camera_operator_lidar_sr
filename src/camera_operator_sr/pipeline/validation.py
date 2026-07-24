"""Validation for pipeline artefacts; failures deliberately make resume rerun."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


RANGE_FILES = ("input_range.npy", "input_intensity.npy", "input_valid.npy", "target_range.npy", "target_valid.npy", "meta.npz")


def frame_names(frame_list: Path) -> list[str]:
    if not frame_list.exists():
        raise FileNotFoundError(frame_list)
    names = [line.strip() for line in frame_list.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f"empty frame manifest: {frame_list}")
    return names


def validate_range_frame(frame: Path) -> None:
    missing = [name for name in RANGE_FILES if not (frame / name).exists()]
    if missing:
        raise FileNotFoundError(frame / missing[0])
    values = {name: np.load(frame / name, allow_pickle=False) for name in RANGE_FILES if name != "meta.npz"}
    meta = np.load(frame / "meta.npz", allow_pickle=False)
    input_shape = values["input_range.npy"].shape
    target_shape = values["target_range.npy"].shape
    if len(input_shape) != 2 or len(target_shape) != 2 or input_shape[1] != target_shape[1]:
        raise ValueError(f"invalid range shapes: {frame}")
    if values["input_intensity.npy"].shape != input_shape or values["input_valid.npy"].shape != input_shape:
        raise ValueError(f"invalid input range companion shape: {frame}")
    if values["target_valid.npy"].shape != target_shape:
        raise ValueError(f"invalid target range companion shape: {frame}")
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"non-finite range output: {frame}")
    for key, shape in (("input_elevation", (input_shape[0],)), ("target_elevation", (target_shape[0],)), ("azimuth", (target_shape[1],))):
        value = meta[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid meta.{key}: {frame}")


def validate_depth_frame(frame: Path) -> None:
    paths = [frame / "relative_depth.npy", frame / "depth_valid.npy"]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(next(path for path in paths if not path.exists()))
    depth, valid = (np.load(path, allow_pickle=False) for path in paths)
    if depth.shape != valid.shape or not depth.size:
        raise ValueError(f"invalid depth shape: {frame}")
    active = valid.astype(bool)
    if not active.any() or not np.isfinite(depth[active]).all():
        raise ValueError(f"invalid depth validity: {frame}")


def validate_csv_metrics(path: Path) -> None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        if not handle or not rows and not (path.read_text().splitlines()[:1]):
            raise ValueError(f"missing CSV header: {path}")
        fields = set(rows[0]) if rows else set(path.read_text().splitlines()[0].split(","))
    if not fields or "empty_group" not in fields:
        raise ValueError(f"invalid metric CSV header: {path}")
    for row in rows:
        if row.get("empty_group") not in {"True", "False", "true", "false", "0", "1", ""}:
            raise ValueError(f"invalid empty_group: {path}")
        for key, value in row.items():
            if value in {"", "None", "null", "nan", "NaN", "inf", "-inf"}:
                continue
            if key.endswith(("count", "ratio", "mae", "rmse", "precision", "recall", "f1", "error", "entropy")):
                try:
                    if not math.isfinite(float(value)):
                        raise ValueError(f"non-finite metric: {path}")
                except ValueError:
                    raise


def validate_inference(path: Path) -> None:
    values = np.load(path, allow_pickle=False)
    required = ("range", "return_probability", "anchor_entropy", "residual")
    if any(key not in values for key in required):
        raise ValueError(f"invalid inference keys: {path}")
    shapes = [values[key].shape for key in required]
    normalized = [tuple(dim for dim in shape if dim != 1) for shape in shapes]
    if not shapes[0] or any(shape != normalized[0] for shape in normalized[1:]) or not all(np.isfinite(values[key]).all() for key in required):
        raise ValueError(f"invalid inference arrays: {path}")
