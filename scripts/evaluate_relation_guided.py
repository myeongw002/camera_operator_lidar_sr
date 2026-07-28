#!/usr/bin/env python3
"""Compare B0, frozen L0, and camera-guided G on identical query masks.

This evaluator is intentionally training/evaluation-only: it loads camera
relative inverse depth, but it never routes a guided checkpoint into the
LiDAR-only deployment path.
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import ProcessedTrainingDataset
from camera_operator_sr.evaluation.beam_metrics import beam_row_metadata, target_row_mask
from camera_operator_sr.evaluation.distance_metrics import DEFAULT_DISTANCE_BINS, gt_distance_bin_mask, parse_distance_bins, serialize_distance_boundary
from camera_operator_sr.evaluation.region_metrics import build_region_masks
from camera_operator_sr.geometry.visibility import build_camera_query_frustum_mask
from camera_operator_sr.models.factory import build_model
from camera_operator_sr.training.checkpoint import load_project_checkpoint, validate_checkpoint_geometry
from camera_operator_sr.training.modules import generated_mask_for


GROUPS = ("full_generated_supported", "camera_guidance_valid", "camera_frustum", "transition", "side", "rear", "front_azimuth")
FIELDS = ("prior_range_mae", "prior_range_rmse", "l0_range_mae", "l0_range_rmse", "guided_range_mae", "guided_range_rmse", "l0_over_prior_mae_improvement", "guided_over_l0_mae_improvement", "guided_over_l0_rmse_improvement", "supported_count", "camera_guidance_count", "camera_guidance_coverage", "mean_valid_camera_candidates", "mean_abs_lidar_correction", "mean_abs_camera_correction", "mean_abs_total_correction", "l0_weight_entropy", "guided_weight_entropy", "anchor_selection_accuracy", "anchor_selection_count", "empty_group")


def move(value, device):
    return value.to(device) if isinstance(value, torch.Tensor) else {key: move(item, device) for key, item in value.items()} if isinstance(value, dict) else value


def shared(value):
    return value[0] if value.ndim == 2 else value


def prior_prediction(output):
    value = (output.lidar.prior_weights * output.lidar.anchor_ranges).sum(-1)
    return torch.where(output.lidar.has_anchor.squeeze(1).bool(), value, torch.zeros_like(value))[:, None]


@dataclass
class Metrics:
    prior_abs: float = 0.0; prior_sq: float = 0.0
    l0_abs: float = 0.0; l0_sq: float = 0.0
    guided_abs: float = 0.0; guided_sq: float = 0.0
    lidar_corr_abs: float = 0.0; camera_corr_abs: float = 0.0; total_corr_abs: float = 0.0
    l0_entropy: float = 0.0; guided_entropy: float = 0.0
    valid_camera_sum: float = 0.0
    supported_count: int = 0; camera_guidance_count: int = 0; both_anchor_count: int = 0
    anchor_correct: int = 0; anchor_count: int = 0

    def update(self, output, target, query):
        supported = query.bool() & target["valid"].bool() & output.lidar.has_anchor.bool()
        selected = supported.bool()
        count = int(selected.sum())
        if not count:
            return
        prior, l0, guided, truth = prior_prediction(output), output.lidar.predicted_range, output.predicted_range, target["range"]
        for prediction, name in ((prior, "prior"), (l0, "l0"), (guided, "guided")):
            error = (prediction - truth)[selected]
            setattr(self, f"{name}_abs", getattr(self, f"{name}_abs") + float(error.abs().sum()))
            setattr(self, f"{name}_sq", getattr(self, f"{name}_sq") + float(error.square().sum()))
        guidance = output.camera_guidance_valid.bool() & selected
        self.supported_count += count; self.camera_guidance_count += int(guidance.sum())
        self.valid_camera_sum += float(output.camera_candidate_valid.sum(-1)[selected.squeeze(1)].sum())
        both = selected.squeeze(1) & output.lidar.anchor_valid.bool().all(-1)
        self.both_anchor_count += int(both.sum())
        self.lidar_corr_abs += float(output.lidar.correction.squeeze(1)[both].abs().sum())
        self.camera_corr_abs += float(output.camera_correction.squeeze(1)[guidance.squeeze(1)].abs().sum())
        self.total_corr_abs += float(output.total_correction.squeeze(1)[both].abs().sum())
        entropy = lambda weights: -(weights * weights.clamp_min(1e-8).log()).sum(-1)
        self.l0_entropy += float(entropy(output.lidar.final_weights)[selected.squeeze(1)].sum())
        self.guided_entropy += float(entropy(output.guided_weights)[selected.squeeze(1)].sum())
        errors = (output.lidar.anchor_ranges - truth.squeeze(1)[..., None]).abs()
        eligible = guidance.squeeze(1) & ~torch.isclose(errors[..., 0], errors[..., 1]) & ~torch.isclose(output.guided_weights[..., 0], output.guided_weights[..., 1])
        self.anchor_correct += int(((output.guided_weights.argmax(-1) == errors.argmin(-1)) & eligible).sum())
        self.anchor_count += int(eligible.sum())

    def result(self):
        if not self.supported_count:
            return {field: (0 if field.endswith("count") else True if field == "empty_group" else None) for field in FIELDS}
        n = self.supported_count
        prior_mae, l0_mae, guided_mae = self.prior_abs / n, self.l0_abs / n, self.guided_abs / n
        l0_rmse, guided_rmse = math.sqrt(self.l0_sq / n), math.sqrt(self.guided_sq / n)
        return {"prior_range_mae": prior_mae, "prior_range_rmse": math.sqrt(self.prior_sq / n), "l0_range_mae": l0_mae, "l0_range_rmse": l0_rmse, "guided_range_mae": guided_mae, "guided_range_rmse": guided_rmse, "l0_over_prior_mae_improvement": prior_mae - l0_mae, "guided_over_l0_mae_improvement": l0_mae - guided_mae, "guided_over_l0_rmse_improvement": l0_rmse - guided_rmse, "supported_count": n, "camera_guidance_count": self.camera_guidance_count, "camera_guidance_coverage": self.camera_guidance_count / n, "mean_valid_camera_candidates": self.valid_camera_sum / n, "mean_abs_lidar_correction": self.lidar_corr_abs / self.both_anchor_count if self.both_anchor_count else None, "mean_abs_camera_correction": self.camera_corr_abs / self.camera_guidance_count if self.camera_guidance_count else None, "mean_abs_total_correction": self.total_corr_abs / self.both_anchor_count if self.both_anchor_count else None, "l0_weight_entropy": self.l0_entropy / n, "guided_weight_entropy": self.guided_entropy / n, "anchor_selection_accuracy": self.anchor_correct / self.anchor_count if self.anchor_count else None, "anchor_selection_count": self.anchor_count, "empty_group": False}


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path); parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--depth-mode", default="correct"); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--distance-bins", nargs="+", default=[str(value) for value in DEFAULT_DISTANCE_BINS])
    args = parser.parse_args()
    checkpoint = load_project_checkpoint(args.checkpoint, map_location=args.device)
    if checkpoint.get("checkpoint_schema_version") != 4 or checkpoint.get("model_config", {}).get("model_type") != "relation_guided":
        raise ValueError("evaluate_relation_guided requires a schema-4 relation_guided checkpoint")
    dataset = ProcessedTrainingDataset(args.dataset_root, split_file=args.split_file, depth_mode=args.depth_mode)
    validate_checkpoint_geometry(checkpoint, dataset[0]); model = build_model(checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval(); bins = parse_distance_bins(args.distance_bins)
    groups, beams, distances, metadata = {}, {}, {}, None
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1, collate_fn=collate_frames):
            batch = move(batch, args.device); output = model(batch); generated = generated_mask_for(batch).bool()
            image_size = tuple(batch["camera"]["relative_depth"].shape[-2:])
            frustum = build_camera_query_frustum_mask(shared(batch["target"]["elevation"]), shared(batch["lidar"]["azimuth"]), batch["calibration"]["K"], batch["calibration"]["T_cam_lidar"], image_size)
            regions = build_region_masks(shared(batch["lidar"]["azimuth"]), camera_frustum=frustum)
            masks = {"full_generated_supported": generated, "camera_guidance_valid": generated & output.camera_guidance_valid.bool()}
            masks.update({name: generated & regions[name].to(device=args.device, dtype=torch.bool) for name in GROUPS if name not in masks})
            for name, mask in masks.items(): groups.setdefault(name, Metrics()).update(output, batch["target"], mask)
            if metadata is None: metadata = beam_row_metadata(shared(batch["lidar"]["elevation"]), shared(batch["target"]["elevation"]))
            for row in range(batch["target"]["range"].shape[-2]):
                row_mask = generated & target_row_mask(1, batch["target"]["range"].shape[-2], batch["target"]["range"].shape[-1], row, device=torch.device(args.device))
                beams.setdefault(row, Metrics()).update(output, batch["target"], row_mask)
            for low, high in zip(bins, bins[1:]):
                for name, mask in masks.items(): distances.setdefault((low, high, name), Metrics()).update(output, batch["target"], mask & gt_distance_bin_mask(batch["target"]["range"], batch["target"]["valid"], low, high))
    args.output_root.mkdir(parents=True, exist_ok=True)
    result = {name: groups.get(name, Metrics()).result() for name in GROUPS}
    summary = {"full_generated_supported": result["full_generated_supported"], "camera_guidance_valid": result["camera_guidance_valid"], "groups": result, "metadata": {"checkpoint": str(args.checkpoint), "model_type": "relation_guided", "depth_mode": args.depth_mode, "comparison_mask": "generated & target_valid & l0.has_anchor & camera_guidance_valid"}}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output_root / "region_metrics.csv", ["region", *FIELDS], [{"region": name, **result[name]} for name in GROUPS])
    write_csv(args.output_root / "beam_metrics.csv", ["target_row", "target_elevation_deg", "is_observed_row", "is_generated_row", *FIELDS], [row | beams.get(int(row["target_row"]), Metrics()).result() for row in metadata])
    distance_rows = [{"distance_min_m": serialize_distance_boundary(low), "distance_max_m": serialize_distance_boundary(high), "region": name, **distances.get((low, high, name), Metrics()).result()} for low, high in zip(bins, bins[1:]) for name in GROUPS]
    write_csv(args.output_root / "distance_metrics.csv", ["distance_min_m", "distance_max_m", "region", *FIELDS], distance_rows)
    write_csv(args.output_root / "guided_relation_metrics.csv", ["group", *FIELDS], [{"group": name, **result[name]} for name in GROUPS])


if __name__ == "__main__":
    main()
