#!/usr/bin/env python3
"""Evaluate a student checkpoint and write global, count-weighted CSV metrics."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import ProcessedTrainingDataset
from camera_operator_sr.data.range_image import range_image_to_pointcloud
from camera_operator_sr.evaluation.beam_metrics import beam_row_metadata, target_row_mask
from camera_operator_sr.evaluation.boundary_metrics import build_range_boundary_mask
from camera_operator_sr.evaluation.distance_metrics import (
    DEFAULT_DISTANCE_BINS,
    gt_distance_bin_mask,
    parse_distance_bins,
    serialize_distance_boundary,
)
from camera_operator_sr.evaluation.evaluator import MetricGroupAccumulator
from camera_operator_sr.evaluation.region_metrics import build_region_masks
from camera_operator_sr.geometry.visibility import (
    build_camera_query_frustum_mask,
    build_gt_visible_valid_mask,
)
from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.training.checkpoint import extract_dataset_geometry, load_project_checkpoint, validate_checkpoint_geometry
from camera_operator_sr.training.modules import generated_mask_for


REGION_ORDER = (
    "full", "camera_frustum", "gt_camera_visible", "camera_boundary",
    "camera_interior", "transition", "side", "rear", "front_azimuth",
)
BEAM_FIELDS = [
    "target_row", "target_elevation_deg", "is_observed_row", "is_generated_row",
    "range_mae", "range_rmse", "return_precision", "return_recall", "return_f1",
    "hallucination_ratio", "missing_ratio", "anchor_mae", "final_mae", "support_error",
    "operator_entropy", "operator_normalized_entropy", "mean_abs_residual",
    "operator_query_count", "all_invalid_positive_query_count",
    "all_invalid_positive_query_ratio", "all_invalid_all_query_count",
    "all_invalid_all_query_ratio", "query_count", "range_count", "empty_group",
]
DISTANCE_FIELDS = [
    "distance_min_m", "distance_max_m", "region", "range_mae", "range_rmse",
    "anchor_mae", "final_mae", "support_error", "operator_entropy",
    "operator_normalized_entropy", "mean_abs_residual", "operator_query_count",
    "all_invalid_positive_query_count", "all_invalid_positive_query_ratio",
    "range_count", "empty_group",
]
OPERATOR_FIELDS = [
    "region", "anchor_mae", "final_mae", "support_error", "operator_entropy",
    "operator_normalized_entropy", "mean_abs_residual", "operator_query_count",
    "all_invalid_positive_query_count", "all_invalid_positive_query_ratio",
    "all_invalid_all_query_count", "all_invalid_all_query_ratio", "empty_group",
]


def move(value, device):
    return value.to(device) if isinstance(value, torch.Tensor) else {key: move(item, device) for key, item in value.items()} if isinstance(value, dict) else value


def _shared_grid(values: torch.Tensor) -> torch.Tensor:
    return values[0] if values.ndim == 2 else values


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path, help="Evaluation frame or sequence list.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--distance-bins", nargs="+", default=[str(value) for value in DEFAULT_DISTANCE_BINS], help="Strictly increasing GT range boundaries; final 'inf' is supported.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    distance_bins = parse_distance_bins(args.distance_bins)
    if not args.split_file.exists():
        raise FileNotFoundError(args.split_file)
    if not any(line.strip() and not line.lstrip().startswith("#") for line in args.split_file.read_text().splitlines()):
        raise RuntimeError("Evaluation split contains no samples.")
    dataset = ProcessedTrainingDataset(args.dataset_root, split_file=args.split_file, depth_mode="none")
    if not len(dataset):
        raise RuntimeError("Evaluation split contains no samples.")
    checkpoint = load_project_checkpoint(args.checkpoint, map_location=args.device)
    validate_checkpoint_geometry(checkpoint, extract_dataset_geometry(dataset[0], candidate_horizontal_radius=int(checkpoint["model_config"]["horizontal_radius"])))
    model = LidarOperatorStudent(**checkpoint["model_config"]).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])

    region_accumulators: dict[str, MetricGroupAccumulator] = {}
    beam_accumulators: dict[int, MetricGroupAccumulator] = {}
    distance_accumulators: dict[tuple[float, float, str], MetricGroupAccumulator] = {}
    beam_metadata: list[dict] | None = None
    regions_seen: set[str] = set()

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1, collate_fn=collate_frames):
            batch = move(batch, args.device)
            validate_checkpoint_geometry(checkpoint, batch)
            output, target = model(batch), batch["target"]
            input_elevation = _shared_grid(batch["lidar"]["elevation"])
            target_elevation = _shared_grid(target["elevation"])
            if beam_metadata is None:
                beam_metadata = beam_row_metadata(input_elevation, target_elevation)

            image_size = batch["calibration"]["image_size"][0].tolist()
            frustum = (
                build_camera_query_frustum_mask(
                    target["elevation"], batch["lidar"]["azimuth"], batch["calibration"]["K"],
                    batch["calibration"]["T_cam_lidar"], tuple(image_size),
                )
                if image_size[0] > 0 and image_size[1] > 0 else None
            )
            azimuth = _shared_grid(batch["lidar"]["azimuth"])
            xyz = range_image_to_pointcloud(target["range"].squeeze(1), target["valid"].squeeze(1), target_elevation, azimuth)
            visible = (
                build_gt_visible_valid_mask(xyz, target["valid"], batch["calibration"]["K"], batch["calibration"]["T_cam_lidar"], tuple(image_size))
                if min(image_size) > 0 else torch.zeros_like(target["valid"])
            )
            boundary = build_range_boundary_mask(target["range"], target["valid"])
            generated = generated_mask_for(batch).bool()
            regions = build_region_masks(
                batch["lidar"]["azimuth"], camera_frustum=frustum,
                gt_camera_visible=visible, camera_boundary=boundary,
            )
            regions_seen.update(regions)

            # Row diagnostics deliberately include observed rows too; their flags
            # are produced with the same elevation mapping as generated training rows.
            for row in range(target["range"].shape[-2]):
                beam_accumulators.setdefault(row, MetricGroupAccumulator()).update(
                    output, target["range"], target["valid"],
                    target_row_mask(target["range"].shape[0], target["range"].shape[-2], target["range"].shape[-1], row, device=args.device),
                )

            for region, region_mask in regions.items():
                evaluation = generated & region_mask.to(device=args.device, dtype=torch.bool)
                region_accumulators.setdefault(region, MetricGroupAccumulator()).update(
                    output, target["range"], target["valid"], evaluation,
                )
                for minimum, maximum in zip(distance_bins, distance_bins[1:]):
                    # This is explicitly GT-range binning.  Invalid targets and
                    # targets outside [minimum, maximum) never enter the group.
                    query = evaluation & gt_distance_bin_mask(target["range"], target["valid"], minimum, maximum)
                    distance_accumulators.setdefault((minimum, maximum, region), MetricGroupAccumulator()).update(
                        output, target["range"], target["valid"], query, include_return=False,
                    )

    assert beam_metadata is not None
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {region: accumulator.csv_result() for region, accumulator in region_accumulators.items()}
    summary["metadata"] = {
        "split_file": str(args.split_file), "num_frames": len(dataset),
        "checkpoint": str(args.checkpoint), "dataset_root": str(args.dataset_root),
        "input_sequences": sorted({path.parent.name for path in dataset.frames}),
        "distance_bins_m": [serialize_distance_boundary(value) for value in distance_bins],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    region_fields = ["region"] + sorted(next((value for key, value in summary.items() if key != "metadata"), {}).keys())
    _write_csv(args.output_root / "region_metrics.csv", region_fields, [
        {"region": region} | summary[region]
        for region in sorted(region_accumulators)
    ])

    beam_rows = []
    for metadata in beam_metadata:
        row = int(metadata["target_row"])
        values = beam_accumulators[row].csv_result()
        beam_rows.append({field: (metadata | values)[field] for field in BEAM_FIELDS})
    _write_csv(args.output_root / "beam_metrics.csv", BEAM_FIELDS, beam_rows)

    distance_rows = []
    for minimum, maximum in zip(distance_bins, distance_bins[1:]):
        for region in REGION_ORDER:
            accumulator = distance_accumulators.get((minimum, maximum, region), MetricGroupAccumulator())
            values = accumulator.csv_result(include_return=False, empty_by_range=True)
            distance_rows.append({
                "distance_min_m": serialize_distance_boundary(minimum),
                "distance_max_m": serialize_distance_boundary(maximum),
                "region": region,
            } | {field: values[field] for field in DISTANCE_FIELDS if field not in {"distance_min_m", "distance_max_m", "region"}})
    _write_csv(args.output_root / "distance_metrics.csv", DISTANCE_FIELDS, distance_rows)

    operator_rows = []
    for region in REGION_ORDER:
        accumulator = region_accumulators.get(region, MetricGroupAccumulator())
        values = accumulator.csv_result()
        operator_rows.append({"region": region} | {field: values[field] for field in OPERATOR_FIELDS if field != "region"})
    _write_csv(args.output_root / "operator_metrics.csv", OPERATOR_FIELDS, operator_rows)


if __name__ == "__main__":
    main()
