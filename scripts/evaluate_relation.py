#!/usr/bin/env python3
"""Evaluate a schema-4 L0 checkpoint against its B0 prior on one common mask."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import ProcessedRangeDataset
from camera_operator_sr.evaluation.beam_metrics import beam_row_metadata, target_row_mask
from camera_operator_sr.evaluation.distance_metrics import DEFAULT_DISTANCE_BINS, gt_distance_bin_mask, parse_distance_bins, serialize_distance_boundary
from camera_operator_sr.evaluation.relation_metrics import RelationMetricAccumulator
from camera_operator_sr.evaluation.region_metrics import build_region_masks
from camera_operator_sr.models.factory import build_model
from camera_operator_sr.training.checkpoint import load_project_checkpoint, validate_checkpoint_geometry
from camera_operator_sr.training.modules import generated_mask_for


REGION_ORDER = ("full", "camera_frustum", "gt_camera_visible", "camera_boundary", "camera_interior", "transition", "side", "rear", "front_azimuth")
METRIC_FIELDS = ["prior_range_mae", "prior_range_rmse", "final_range_mae", "final_range_rmse", "mae_improvement", "rmse_improvement", "supported_count", "valid_target_count", "unsupported_valid_target_count", "anchor_coverage", "mean_abs_correction", "both_anchor_count", "prior_weight_entropy", "final_weight_entropy", "anchor_selection_accuracy", "anchor_selection_count", "empty_group"]
BEAM_FIELDS = ["target_row", "target_elevation_deg", "is_observed_row", "is_generated_row", *METRIC_FIELDS]
REGION_FIELDS = ["region", *METRIC_FIELDS]
DISTANCE_FIELDS = ["distance_min_m", "distance_max_m", "region", *METRIC_FIELDS]


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
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--distance-bins", nargs="+", default=[str(value) for value in DEFAULT_DISTANCE_BINS])
    args = parser.parse_args()
    if not args.split_file.exists():
        raise FileNotFoundError(args.split_file)
    checkpoint = load_project_checkpoint(args.checkpoint, map_location=args.device)
    if checkpoint.get("model_config", {}).get("model_type") != "relation_l0":
        raise ValueError("evaluate_relation requires a relation_l0 schema-4 checkpoint; legacy checkpoints are unsupported")
    if checkpoint.get("checkpoint_schema_version") != 4:
        raise ValueError("evaluate_relation requires checkpoint_schema_version 4")
    dataset = ProcessedRangeDataset(args.dataset_root, split_file=args.split_file, depth_mode="none")
    if not len(dataset):
        raise RuntimeError("Evaluation split contains no samples.")
    model = build_model(checkpoint["model_config"]).to(args.device)
    validate_checkpoint_geometry(checkpoint, dataset[0])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    bins = parse_distance_bins(args.distance_bins)
    global_accumulator = RelationMetricAccumulator()
    region_accumulators: dict[str, RelationMetricAccumulator] = {}
    beam_accumulators: dict[int, RelationMetricAccumulator] = {}
    distance_accumulators: dict[tuple[float, float, str], RelationMetricAccumulator] = {}
    beam_metadata = None
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1, collate_fn=collate_frames):
            batch = move(batch, args.device)
            output, target = model(batch), batch["target"]
            generated = generated_mask_for(batch).bool()
            azimuth = _shared_grid(batch["lidar"]["azimuth"])
            if beam_metadata is None:
                beam_metadata = beam_row_metadata(_shared_grid(batch["lidar"]["elevation"]), _shared_grid(target["elevation"]))
            regions = build_region_masks(azimuth)
            global_accumulator.update(output, target["range"], target["valid"], generated)
            for row in range(target["range"].shape[-2]):
                mask = generated & target_row_mask(target["range"].shape[0], target["range"].shape[-2], target["range"].shape[-1], row, device=torch.device(args.device))
                beam_accumulators.setdefault(row, RelationMetricAccumulator()).update(output, target["range"], target["valid"], mask)
            for region, region_mask in regions.items():
                evaluation = generated & region_mask.to(device=args.device, dtype=torch.bool)
                region_accumulators.setdefault(region, RelationMetricAccumulator()).update(output, target["range"], target["valid"], evaluation)
                for minimum, maximum in zip(bins, bins[1:]):
                    query = evaluation & gt_distance_bin_mask(target["range"], target["valid"], minimum, maximum)
                    distance_accumulators.setdefault((minimum, maximum, region), RelationMetricAccumulator()).update(output, target["range"], target["valid"], query)
    assert beam_metadata is not None
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {"global": global_accumulator.result()} | {region: region_accumulators.get(region, RelationMetricAccumulator()).result() for region in REGION_ORDER}
    summary["metadata"] = {"checkpoint": str(args.checkpoint), "checkpoint_schema_version": checkpoint["checkpoint_schema_version"], "model_type": "relation_l0", "dataset_root": str(args.dataset_root), "split_file": str(args.split_file), "num_frames": len(dataset), "model_config": checkpoint["model_config"], "distance_bins_m": [serialize_distance_boundary(value) for value in bins], "available_metrics": METRIC_FIELDS, "unavailable_metrics": ["return", "residual", "operator"]}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(args.output_root / "region_metrics.csv", REGION_FIELDS, [{"region": region} | region_accumulators.get(region, RelationMetricAccumulator()).result() for region in REGION_ORDER])
    _write_csv(args.output_root / "beam_metrics.csv", BEAM_FIELDS, [metadata | beam_accumulators.get(int(metadata["target_row"]), RelationMetricAccumulator()).result() for metadata in beam_metadata])
    distance_rows = []
    for minimum, maximum in zip(bins, bins[1:]):
        for region in REGION_ORDER:
            distance_rows.append({"distance_min_m": serialize_distance_boundary(minimum), "distance_max_m": serialize_distance_boundary(maximum), "region": region} | distance_accumulators.get((minimum, maximum, region), RelationMetricAccumulator()).result())
    _write_csv(args.output_root / "distance_metrics.csv", DISTANCE_FIELDS, distance_rows)
    _write_csv(args.output_root / "relation_metrics.csv", ["group", *METRIC_FIELDS], [{"group": "global"} | global_accumulator.result()])


if __name__ == "__main__":
    main()
