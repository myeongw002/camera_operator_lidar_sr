#!/usr/bin/env python3
"""Evaluate the checkpoint-free B0 geometric prior with range metrics only."""

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
from camera_operator_sr.evaluation.evaluator import RangeAccumulator
from camera_operator_sr.evaluation.region_metrics import build_region_masks
from camera_operator_sr.geometry.visibility import (
    build_camera_query_frustum_mask,
    build_gt_visible_valid_mask,
)
from camera_operator_sr.models.relation import GeometricBaselineModel
from camera_operator_sr.training.modules import generated_mask_for


REGION_ORDER = (
    "full", "camera_frustum", "gt_camera_visible", "camera_boundary",
    "camera_interior", "transition", "side", "rear", "front_azimuth",
)
BEAM_FIELDS = [
    "target_row", "target_elevation_deg", "is_observed_row", "is_generated_row",
    "range_mae", "range_rmse", "range_count",
]
DISTANCE_FIELDS = ["distance_min_m", "distance_max_m", "region", "range_mae", "range_rmse", "range_count", "empty_group"]
REGION_FIELDS = ["region", "range_mae", "range_rmse", "range_count", "empty_group"]


def move(value, device):
    return value.to(device) if isinstance(value, torch.Tensor) else {key: move(item, device) for key, item in value.items()} if isinstance(value, dict) else value


def _shared_grid(values: torch.Tensor) -> torch.Tensor:
    return values[0] if values.ndim == 2 else values


def _range_result(accumulator: RangeAccumulator) -> dict:
    values = accumulator.result()
    values["empty_group"] = accumulator.count == 0
    if accumulator.count == 0:
        values["range_mae"] = values["range_rmse"] = None
    return values


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path, help="Evaluation frame or sequence list.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--horizontal-radius", type=int, default=1)
    parser.add_argument("--distance-bins", nargs="+", default=[str(value) for value in DEFAULT_DISTANCE_BINS])
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
    model = GeometricBaselineModel(horizontal_radius=args.horizontal_radius).to(args.device).eval()
    region_accumulators: dict[str, RangeAccumulator] = {}
    beam_accumulators: dict[int, RangeAccumulator] = {}
    distance_accumulators: dict[tuple[float, float, str], RangeAccumulator] = {}
    beam_metadata: list[dict] | None = None

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1, collate_fn=collate_frames):
            batch = move(batch, args.device)
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
                ) if min(image_size) > 0 else None
            )
            azimuth = _shared_grid(batch["lidar"]["azimuth"])
            xyz = range_image_to_pointcloud(target["range"].squeeze(1), target["valid"].squeeze(1), target_elevation, azimuth)
            visible = (
                build_gt_visible_valid_mask(xyz, target["valid"], batch["calibration"]["K"], batch["calibration"]["T_cam_lidar"], tuple(image_size))
                if min(image_size) > 0 else torch.zeros_like(target["valid"])
            )
            regions = build_region_masks(
                batch["lidar"]["azimuth"], camera_frustum=frustum, gt_camera_visible=visible,
                camera_boundary=build_range_boundary_mask(target["range"], target["valid"]),
            )
            generated = generated_mask_for(batch).bool()
            valid_generated = generated & target["valid"].bool()
            for row in range(target["range"].shape[-2]):
                mask = generated & target_row_mask(target["range"].shape[0], target["range"].shape[-2], target["range"].shape[-1], row, device=args.device)
                beam_accumulators.setdefault(row, RangeAccumulator()).update(output.predicted_range, target["range"], mask & target["valid"].bool())
            for region, region_mask in regions.items():
                evaluation = valid_generated & region_mask.to(device=args.device, dtype=torch.bool)
                region_accumulators.setdefault(region, RangeAccumulator()).update(output.predicted_range, target["range"], evaluation)
                for minimum, maximum in zip(distance_bins, distance_bins[1:]):
                    query = evaluation & gt_distance_bin_mask(target["range"], target["valid"], minimum, maximum)
                    distance_accumulators.setdefault((minimum, maximum, region), RangeAccumulator()).update(output.predicted_range, target["range"], query)

    assert beam_metadata is not None
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {region: _range_result(accumulator) for region, accumulator in region_accumulators.items()}
    summary["metadata"] = {
        "model_type": "geometric_prior", "checkpoint": None, "split_file": str(args.split_file),
        "num_frames": len(dataset), "dataset_root": str(args.dataset_root),
        "horizontal_radius": args.horizontal_radius,
        "distance_bins_m": [serialize_distance_boundary(value) for value in distance_bins],
        "available_metrics": ["range_mae", "range_rmse", "range_count"],
        "unavailable_metrics": ["return", "residual", "operator"],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(args.output_root / "region_metrics.csv", REGION_FIELDS, [
        {"region": region} | _range_result(region_accumulators.get(region, RangeAccumulator()))
        for region in REGION_ORDER
    ])
    _write_csv(args.output_root / "beam_metrics.csv", BEAM_FIELDS, [
        metadata | {field: _range_result(beam_accumulators.get(int(metadata["target_row"]), RangeAccumulator()))[field] for field in ("range_mae", "range_rmse", "range_count")}
        for metadata in beam_metadata
    ])
    distance_rows = []
    for minimum, maximum in zip(distance_bins, distance_bins[1:]):
        for region in REGION_ORDER:
            values = _range_result(distance_accumulators.get((minimum, maximum, region), RangeAccumulator()))
            distance_rows.append({
                "distance_min_m": serialize_distance_boundary(minimum),
                "distance_max_m": serialize_distance_boundary(maximum), "region": region,
            } | values)
    _write_csv(args.output_root / "distance_metrics.csv", DISTANCE_FIELDS, distance_rows)


if __name__ == "__main__":
    main()
