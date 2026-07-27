#!/usr/bin/env python3
"""Run a student checkpoint on one processed frame and preserve observed rows."""
import argparse
import os
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
    debug = os.environ.get("CAMERA_OPERATOR_SR_DEBUG") == "1"
    if debug: print(f"[debug:infer] frame_load_start frame={args.frame_root}", flush=True)
    batch = collate_frames([LidarInferenceDataset(args.frame_root.parent, [args.frame_root])[0]])
    batch = {key: {subkey: value.to(args.device) if isinstance(value, torch.Tensor) else value for subkey, value in part.items()} if isinstance(part, dict) else part for key, part in batch.items()}
    checkpoint = load_project_checkpoint(args.checkpoint, map_location=args.device)
    if debug: print(f"[debug:infer] checkpoint_loaded checkpoint={args.checkpoint}", flush=True)
    model = LidarOperatorStudent(**checkpoint["model_config"]).to(args.device).eval()
    validate_checkpoint_geometry(checkpoint, batch)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        if debug: print("[debug:infer] model_forward_start", flush=True)
        output = model(batch)
        ranges, probability = fuse_observed_rows(output, batch)
        if debug: print("[debug:infer] model_forward_done", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if debug: print(f"[debug:infer] save_start output={args.output}", flush=True)
    np.savez_compressed(args.output, range=ranges.cpu().numpy(), return_probability=probability.cpu().numpy(), anchor_entropy=(-(output.anchor_weights * torch.log(output.anchor_weights.clamp_min(1e-8))).sum(-1)).cpu().numpy(), residual=output.residual.cpu().numpy())
    if debug: print("[debug:infer] save_done", flush=True)


if __name__ == "__main__":
    main()
