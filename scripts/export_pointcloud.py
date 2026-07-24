#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

from camera_operator_sr.data.range_image import range_image_to_pointcloud


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, type=Path, help=".npz emitted by infer.py")
    parser.add_argument("--target-elevation", required=True, type=Path, help=".npy radians [64]")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--return-threshold", type=float, default=0.5)
    args = parser.parse_args()
    data = np.load(args.prediction)
    ranges = torch.from_numpy(data["range"]).squeeze(0).squeeze(0)
    valid = torch.from_numpy(data["return_probability"]).squeeze(0).squeeze(0).gt(args.return_threshold)
    elevation = torch.from_numpy(np.load(args.target_elevation)).to(ranges.dtype)
    azimuth = torch.arange(ranges.shape[-1], dtype=ranges.dtype) * (2 * torch.pi / ranges.shape[-1]) - torch.pi + torch.pi / ranges.shape[-1]
    points = range_image_to_pointcloud(ranges, valid, elevation, azimuth)[valid].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, points, fmt="%.6f", header="x y z")


if __name__ == "__main__":
    main()
