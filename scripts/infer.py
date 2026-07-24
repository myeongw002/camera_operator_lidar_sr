#!/usr/bin/env python3
"""Run a student checkpoint on one processed frame and preserve observed rows."""
import argparse
from pathlib import Path

import numpy as np
import torch

from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import LidarInferenceDataset
from camera_operator_sr.inference import fuse_observed_rows
from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.training.checkpoint import load_project_checkpoint, validate_checkpoint_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frame-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    batch = collate_frames([LidarInferenceDataset(args.frame_root.parent, [args.frame_root])[0]])
    batch = {key: {subkey: value.to(args.device) if isinstance(value, torch.Tensor) else value for subkey, value in part.items()} if isinstance(part, dict) else part for key, part in batch.items()}
    checkpoint = load_project_checkpoint(args.checkpoint, map_location=args.device)
    model = LidarOperatorStudent(**checkpoint["model_config"]).to(args.device).eval()
    validate_checkpoint_geometry(checkpoint, batch)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        output = model(batch)
        ranges, probability = fuse_observed_rows(output, batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, range=ranges.cpu().numpy(), return_probability=probability.cpu().numpy(), anchor_entropy=(-(output.anchor_weights * torch.log(output.anchor_weights.clamp_min(1e-8))).sum(-1)).cpu().numpy(), residual=output.residual.cpu().numpy())


if __name__ == "__main__":
    main()
