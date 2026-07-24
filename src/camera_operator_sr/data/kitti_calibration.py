from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KittiCalibration:
    K: np.ndarray
    T_cam_lidar: np.ndarray
    image_size: tuple[int, int]


def _read_entries(path: Path) -> dict[str, np.ndarray]:
    entries: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            entries[key.strip()] = np.fromstring(raw, sep=" ", dtype=np.float64)
        except ValueError:
            continue
    return entries


def _first(entries: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in entries and entries[name].size:
            return entries[name]
    raise KeyError(f"calibration is missing one of: {', '.join(names)}")


def load_kitti_calibration(calibration_path: str | Path, camera_id: int = 2) -> KittiCalibration:
    """Load object-detection or odometry calibration into camera<-LiDAR form.

    KITTI Odometry calib.txt commonly contains P0…P3 and ``Tr`` but no
    R0_rect/R_rect_00.  Its projection matrices are already the camera-space
    contract used here, so missing rectification is intentionally identity.
    """
    path = Path(calibration_path)
    entries = _read_entries(path)
    projection = _first(entries, (f"P{camera_id}", f"P_rect_0{camera_id}"))
    if projection.size != 12:
        raise ValueError("camera projection must contain 12 values")
    projection = projection.reshape(3, 4)
    rectification = next((entries[name] for name in ("R0_rect", "R_rect_00") if name in entries and entries[name].size), np.eye(3, dtype=np.float64).reshape(-1))
    if rectification.size != 9:
        raise ValueError("rectification matrix must contain 9 values")
    velo = _first(entries, ("Tr_velo_to_cam", "Tr_velo_cam", "Tr"))
    if velo.size != 12:
        raise ValueError("Velodyne transform must contain 12 values")
    rect4 = np.eye(4, dtype=np.float64)
    rect4[:3, :3] = rectification.reshape(3, 3)
    velo4 = np.eye(4, dtype=np.float64)
    velo4[:3, :] = velo.reshape(3, 4)
    K = projection[:, :3]
    if abs(np.linalg.det(K)) < 1e-12:
        raise ValueError("projection intrinsic block is singular")
    camera_offset = np.eye(4, dtype=np.float64)
    camera_offset[:3, 3] = np.linalg.solve(K, projection[:, 3])
    size_values = entries.get(f"S_{camera_id:02d}", entries.get(f"S_rect_0{camera_id}"))
    image_size = (int(size_values[1]), int(size_values[0])) if size_values is not None and size_values.size >= 2 else (0, 0)
    return KittiCalibration(K=K.astype(np.float32), T_cam_lidar=(camera_offset @ rect4 @ velo4).astype(np.float32), image_size=image_size)
