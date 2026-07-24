#!/usr/bin/env python3
"""Save an observed-LiDAR-on-camera projection diagnostic PNG."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from camera_operator_sr.data.range_image import range_image_to_pointcloud
from camera_operator_sr.geometry.projection import project_lidar_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-root", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    meta = np.load(args.frame_root / "meta.npz")
    ranges = torch.as_tensor(np.load(args.frame_root / "input_range.npy"), dtype=torch.float32)[None]
    valid = torch.as_tensor(np.load(args.frame_root / "input_valid.npy"), dtype=torch.float32)[None]
    xyz = range_image_to_pointcloud(ranges, valid, torch.as_tensor(meta["input_elevation"]), torch.as_tensor(meta["azimuth"])).reshape(1, -1, 3)
    projection = project_lidar_points(xyz, torch.as_tensor(meta["K"])[None], torch.as_tensor(meta["T_cam_lidar"])[None], image.shape[:2])
    flat_range = ranges.reshape(-1).numpy()
    inside = projection.valid[0].cpu().numpy()
    uv = projection.uv[0].cpu().numpy()
    for point, distance in zip(uv[inside], flat_range[inside]):
        color = cv2.applyColorMap(np.array([[min(255, int(distance / 80 * 255))]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0].tolist()
        cv2.circle(image, tuple(np.round(point).astype(int)), 2, color, -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image)
    behind = projection.camera_depth[0].le(0).cpu().numpy()
    report = {"projected": int(inside.sum()), "total": int(inside.size), "inside_ratio": float(inside.mean()), "behind_camera": int(behind.sum()), "outside_or_invalid": int((~inside & ~behind).sum())}
    if report["inside_ratio"] < 0.01:
        print("WARNING: very few LiDAR points project into the image; verify calibration direction and camera id.")
    print(report)


if __name__ == "__main__":
    main()
