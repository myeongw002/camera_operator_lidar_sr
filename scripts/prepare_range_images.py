#!/usr/bin/env python3
"""Convert KITTI-style float32 x/y/z/intensity scans to synthetic 16/64 ranges."""
import argparse
from pathlib import Path

import numpy as np
import torch

from camera_operator_sr.data.kitti_calibration import load_kitti_calibration
from camera_operator_sr.data.range_image import pointcloud_to_range_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", "--pointcloud-root", dest="scan_root", required=True, type=Path, help="directory containing *.bin scans")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-elevation", required=True, type=Path, help=".npy radians [64]")
    parser.add_argument("--input-row-indices", required=True, help="comma-separated target row indices")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--row-tolerance", type=float, default=0.003)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--camera-id", type=int, default=2)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--allow-identity-calibration", action="store_true")
    parser.add_argument("--frame-list", type=Path, help="newline-delimited scan stems selected by the pipeline")
    args = parser.parse_args()
    elevation = torch.from_numpy(np.load(args.target_elevation)).float()
    indices = np.fromstring(args.input_row_indices, sep=",", dtype=np.int64)
    if elevation.numel() != 64 or len(indices) != 16 or len(np.unique(indices)) != 16:
        raise ValueError("provide 64 target elevations and 16 unique input indices")
    azimuth = np.arange(args.width, dtype=np.float32) * (2 * np.pi / args.width) - np.pi + np.pi / args.width
    if args.calibration_root is None and not args.allow_identity_calibration:
        raise ValueError("--calibration-root is required unless --allow-identity-calibration is explicitly set")
    selected = None if args.frame_list is None else {line.strip() for line in args.frame_list.read_text().splitlines() if line.strip()}
    if selected is not None and not selected:
        raise ValueError("--frame-list contains no frames")
    for scan_path in sorted(args.scan_root.glob("*.bin")):
        if selected is not None and scan_path.stem not in selected:
            continue
        if args.calibration_root is None:
            K, T, image_size = np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32), (0, 0)
        else:
            per_frame = args.calibration_root / f"{scan_path.stem}.txt"
            calibration_path = per_frame if per_frame.exists() else args.calibration_root / "calib.txt"
            if not calibration_path.exists():
                raise FileNotFoundError(f"no calibration for {scan_path}: expected {per_frame} or {calibration_path}")
            calibration = load_kitti_calibration(calibration_path, args.camera_id)
            K, T, image_size = calibration.K, calibration.T_cam_lidar, calibration.image_size
        if args.image_root is not None:
            image_path = next((path for path in (args.image_root / f"{scan_path.stem}.png", args.image_root / f"{scan_path.stem}.jpg") if path.exists()), None)
            if image_path is None:
                raise FileNotFoundError(f"image missing for {scan_path.stem}")
            from PIL import Image
            with Image.open(image_path) as image:
                image_size = (image.height, image.width)
        raw = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        image = pointcloud_to_range_image(torch.from_numpy(raw[:, :3]), torch.from_numpy(raw[:, 3]), elevation, args.width, row_tolerance=args.row_tolerance)
        frame = args.output_root / scan_path.stem
        frame.mkdir(parents=True, exist_ok=True)
        np.save(frame / "target_range.npy", image.range.squeeze(0).numpy())
        np.save(frame / "target_intensity.npy", image.intensity.squeeze(0).numpy())
        np.save(frame / "target_valid.npy", image.valid.squeeze(0).numpy())
        input_indices = torch.from_numpy(indices)
        np.save(frame / "input_range.npy", image.range.squeeze(0)[input_indices].numpy())
        np.save(frame / "input_intensity.npy", image.intensity.squeeze(0)[input_indices].numpy())
        np.save(frame / "input_valid.npy", image.valid.squeeze(0)[input_indices].numpy())
        np.savez_compressed(frame / "meta.npz", input_elevation=elevation[indices].numpy(), target_elevation=elevation.numpy(), azimuth=azimuth, K=K, T_cam_lidar=T, image_size=np.asarray(image_size, dtype=np.int32))


if __name__ == "__main__":
    main()
