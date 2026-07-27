"""Versioned checkpoint metadata and fail-fast compatibility checks."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import torch
from torch import nn

from camera_operator_sr.geometry.candidate_graph import expected_candidate_count
from camera_operator_sr.geometry.validation import assert_shared_geometry


CHECKPOINT_SCHEMA_VERSION = 4
LEGACY_CHECKPOINT_SCHEMA_VERSION = 3
GEOMETRY_KEYS = (
    "input_elevation", "target_elevation", "azimuth", "width",
    "input_beam_count", "target_beam_count", "candidate_horizontal_radius",
    "candidate_count",
)


def load_project_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    """Load a trusted project checkpoint, including schema-3 NumPy RNG state."""
    return torch.load(path, map_location=map_location, weights_only=False)


@dataclass(frozen=True)
class GeometryMetadata:
    input_elevation: list[float]
    target_elevation: list[float]
    azimuth: list[float]
    width: int
    input_beam_count: int
    target_beam_count: int
    candidate_horizontal_radius: int
    candidate_count: int
    azimuth_direction: str = "increasing"
    azimuth_origin_rad: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _one_geometry(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim == 2:
        assert_shared_geometry(values, name)
        values = values[0]
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after batch selection")
    return values.detach().cpu()


def build_geometry_metadata(*, input_elevation: torch.Tensor, target_elevation: torch.Tensor, azimuth: torch.Tensor, candidate_horizontal_radius: int, candidate_count: int | None = None) -> dict:
    input_elevation = _one_geometry(input_elevation, "input elevation")
    target_elevation = _one_geometry(target_elevation, "target elevation")
    azimuth = _one_geometry(azimuth, "azimuth")
    if azimuth.numel() == 0 or input_elevation.numel() == 0 or target_elevation.numel() == 0:
        raise ValueError("Geometry metadata requires non-empty elevation and azimuth grids")
    expected = expected_candidate_count(candidate_horizontal_radius)
    candidate_count = expected if candidate_count is None else int(candidate_count)
    if candidate_count != expected:
        raise ValueError(f"candidate_count={candidate_count} does not match horizontal radius {candidate_horizontal_radius} (expected {expected})")
    return GeometryMetadata(
        input_elevation=input_elevation.tolist(), target_elevation=target_elevation.tolist(), azimuth=azimuth.tolist(),
        width=int(azimuth.numel()), input_beam_count=int(input_elevation.numel()),
        target_beam_count=int(target_elevation.numel()), candidate_horizontal_radius=int(candidate_horizontal_radius),
        candidate_count=candidate_count, azimuth_direction="increasing" if azimuth.numel() < 2 or float(azimuth[-1] - azimuth[0]) > 0 else "decreasing",
        azimuth_origin_rad=float(azimuth[0]),
    ).as_dict()


def extract_dataset_geometry(sample_or_batch: dict, *, candidate_horizontal_radius: int, candidate_count: int | None = None) -> GeometryMetadata:
    geometry = build_geometry_metadata(
        input_elevation=sample_or_batch["lidar"]["elevation"], target_elevation=sample_or_batch["target"]["elevation"],
        azimuth=sample_or_batch["lidar"]["azimuth"], candidate_horizontal_radius=candidate_horizontal_radius,
        candidate_count=candidate_count,
    )
    return GeometryMetadata(**geometry)


def geometry_from_sample(sample: dict, model: nn.Module) -> dict:
    radius = int(getattr(model, "horizontal_radius", getattr(model, "model_config", {}).get("horizontal_radius", 1)))
    geometry = build_geometry_metadata(input_elevation=sample["lidar"]["elevation"], target_elevation=sample["target"]["elevation"], azimuth=sample["lidar"]["azimuth"], candidate_horizontal_radius=radius)
    if getattr(model, "model_type", None) == "relation_l0":
        geometry.update(candidate_layout="lower[-1,0,+1],upper[-1,0,+1]", anchor_slots=[1, 4])
    return geometry


def save_checkpoint(path: str | Path, model: nn.Module, *, epoch: int, global_step: int, sample: dict, optimizer=None, data_config: dict | None = None, operator_config: dict | None = None, loss_config: dict | None = None, dataset_split: str | None = None, depth_mode: str = "none", validation_score: float | None = None, validation_count: int | None = None, experiment_metadata: dict | None = None, advantage_config: dict | None = None, rng_state: dict | None = None, best_validation_score: float | None = None, best_epoch: int | None = None, best_global_step: int | None = None, source_checkpoints: dict | None = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    geometry = geometry_from_sample(sample, model)
    schema = CHECKPOINT_SCHEMA_VERSION if getattr(model, "model_type", None) == "relation_l0" else LEGACY_CHECKPOINT_SCHEMA_VERSION
    payload = {"checkpoint_schema_version": schema, "geometry": geometry, "model": model.state_dict(), "model_config": getattr(model, "model_config", {}), "data_config": data_config or {}, "operator_config": operator_config or {}, "loss_config": loss_config or {}, "dataset_split": dataset_split, "depth_mode": str(depth_mode), "epoch": epoch, "global_step": global_step, "rng_state": rng_state, "source_checkpoints": source_checkpoints or {}}
    if optimizer is not None: payload["optimizer"] = optimizer.state_dict()
    if validation_score is not None or validation_count is not None:
        if validation_score is None or validation_count is None or validation_count == 0:
            raise ValueError("best checkpoint requires non-zero validation_count and validation_score")
        payload.update(validation_metric="global_query_weighted_range_mae", validation_score=float(validation_score), validation_count=int(validation_count))
        payload.update(current_validation_score=float(validation_score), best_validation_score=float(best_validation_score if best_validation_score is not None else validation_score), best_epoch=int(best_epoch if best_epoch is not None else epoch), best_global_step=int(best_global_step if best_global_step is not None else global_step))
    elif experiment_metadata is not None:
        # A train-only run remains resumable; absence of validation is explicit, never implied by a missing key.
        payload.update(validation_metric="global_query_weighted_range_mae", validation_score=None, validation_count=0,
                       current_validation_score=None, best_validation_score=float(best_validation_score if best_validation_score is not None else float("inf")),
                       best_epoch=int(best_epoch or 0), best_global_step=int(best_global_step or 0))
    if experiment_metadata is not None:
        experiment_config = experiment_metadata.get("experiment_config", experiment_metadata["run_config"])
        current_invocation = experiment_metadata.get("current_invocation", experiment_metadata["run_config"])
        payload.update(experiment_name=experiment_metadata["experiment_name"], experiment_type=experiment_metadata["experiment_type"], seed=int(experiment_metadata["seed"]), deterministic=bool(experiment_metadata["deterministic"]), run_config=experiment_config, current_invocation=current_invocation, experiment_config=experiment_config, invocation_index=int(current_invocation.get("invocation_index", experiment_metadata.get("invocation_index", 0))), manifest=experiment_metadata["manifest"], split_hashes={key: value["sha256"] if value else None for key, value in experiment_metadata["manifest"]["splits"].items()})
        payload["source_checkpoints"] = source_checkpoints or experiment_metadata["manifest"].get("source_checkpoints", {})
    if advantage_config is not None: payload["advantage_config"] = advantage_config
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def _checkpoint_geometry(checkpoint: dict) -> dict:
    schema = checkpoint.get("checkpoint_schema_version")
    if schema not in {LEGACY_CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_SCHEMA_VERSION} or "geometry" not in checkpoint:
        raise ValueError("Checkpoint does not contain geometry metadata. This checkpoint predates schema version 2.")
    geometry = checkpoint["geometry"]
    missing = [key for key in GEOMETRY_KEYS if key not in geometry]
    if missing: raise ValueError(f"Checkpoint geometry metadata missing: {', '.join(missing)}")
    expected = expected_candidate_count(int(geometry["candidate_horizontal_radius"]))
    if int(geometry["candidate_count"]) != expected:
        raise ValueError("Checkpoint geometry metadata has inconsistent candidate_count")
    if int(geometry["width"]) != len(geometry["azimuth"]) or int(geometry["input_beam_count"]) != len(geometry["input_elevation"]) or int(geometry["target_beam_count"]) != len(geometry["target_elevation"]):
        raise ValueError("Checkpoint geometry metadata has inconsistent width or beam counts")
    if schema == CHECKPOINT_SCHEMA_VERSION:
        if geometry.get("candidate_layout") != "lower[-1,0,+1],upper[-1,0,+1]" or geometry.get("anchor_slots") != [1, 4]:
            raise ValueError("Relation checkpoint geometry metadata has incompatible candidate layout")
    return geometry


def _geometry_dict(value: GeometryMetadata | dict) -> dict:
    geometry = value.as_dict() if isinstance(value, GeometryMetadata) else value
    missing = [key for key in GEOMETRY_KEYS if key not in geometry]
    if missing: raise ValueError(f"Dataset geometry metadata missing: {', '.join(missing)}")
    if int(geometry["width"]) != len(geometry["azimuth"]) or int(geometry["input_beam_count"]) != len(geometry["input_elevation"]) or int(geometry["target_beam_count"]) != len(geometry["target_elevation"]):
        raise ValueError("Dataset geometry metadata has inconsistent width or beam counts")
    if int(geometry["candidate_count"]) != expected_candidate_count(int(geometry["candidate_horizontal_radius"])):
        raise ValueError("Dataset geometry metadata has inconsistent candidate_count")
    return geometry


def _different(left: object, right: object, name: str, atol: float) -> bool:
    if name in {"input_elevation", "target_elevation", "azimuth"}:
        a, b = torch.as_tensor(left), torch.as_tensor(right)
        return a.shape != b.shape or not torch.allclose(a, b, atol=atol, rtol=0.0)
    return int(left) != int(right)


def validate_checkpoint_geometry(checkpoint: dict, dataset_geometry: GeometryMetadata | dict, *, atol: float = 1e-6) -> None:
    checkpoint_geometry = _checkpoint_geometry(checkpoint)
    if isinstance(dataset_geometry, dict) and "lidar" in dataset_geometry:
        dataset_geometry = extract_dataset_geometry(dataset_geometry, candidate_horizontal_radius=int(checkpoint_geometry["candidate_horizontal_radius"]), candidate_count=int(checkpoint_geometry["candidate_count"]))
    dataset_geometry = _geometry_dict(dataset_geometry)
    mismatches = [key for key in GEOMETRY_KEYS if _different(checkpoint_geometry[key], dataset_geometry[key], key, atol)]
    if mismatches:
        key = mismatches[0]
        raise ValueError(f"Checkpoint geometry mismatch: {key}\ncheckpoint: {checkpoint_geometry[key]}\ndataset: {dataset_geometry[key]}")


def validate_checkpoint_pair(left_checkpoint: dict, right_checkpoint: dict, *, left_name: str, right_name: str, atol: float = 1e-6) -> None:
    left, right = _checkpoint_geometry(left_checkpoint), _checkpoint_geometry(right_checkpoint)
    for key in GEOMETRY_KEYS:
        if _different(left[key], right[key], key, atol):
            raise ValueError(f"Checkpoint pair mismatch ({left_name} vs {right_name}): {key}")
    shared_config = ("lidar_feature_dim", "hidden_dim", "horizontal_radius", "residual_scale_alpha", "residual_min", "residual_max")
    lconfig, rconfig = left_checkpoint.get("model_config", {}), right_checkpoint.get("model_config", {})
    for key in shared_config:
        if lconfig.get(key) != rconfig.get(key):
            raise ValueError(f"Checkpoint pair model config mismatch ({left_name} vs {right_name}): {key}")


def validate_teacher_depth_mode(checkpoint: dict, requested_depth_mode: str, *, allow_mismatch: bool = False) -> None:
    if "depth_mode" not in checkpoint: raise ValueError("Teacher checkpoint does not contain depth_mode metadata")
    if checkpoint["depth_mode"] != requested_depth_mode:
        message = f"Teacher depth mode mismatch\ncheckpoint depth mode: {checkpoint['depth_mode']}\nrequested depth mode: {requested_depth_mode}"
        if not allow_mismatch: raise ValueError(message)
        warnings.warn(message + "\nThe mismatch was explicitly allowed by the user.", UserWarning, stacklevel=2)
