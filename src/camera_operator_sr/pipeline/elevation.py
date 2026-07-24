"""Data-derived fallback for a missing KITTI HDL-64 elevation table."""
from __future__ import annotations

import numpy as np
from pathlib import Path


def estimate_elevations(scan_paths: list[Path], *, beam_count: int = 64, max_points: int = 500_000) -> np.ndarray:
    """Estimate ordered laser elevations from real x/y/z scan samples.

    HDL scans contain repeated vertical-angle bands.  One-dimensional Lloyd
    clustering preserves those bands without pretending that a linear table is
    calibrated data.  The result is an estimate and is recorded as such by the
    pipeline provenance file.
    """
    values: list[np.ndarray] = []
    remaining = max_points
    for path in scan_paths:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"invalid KITTI scan (not float32 x/y/z/intensity): {path}")
        xyz = raw.reshape(-1, 4)[:, :3]
        horizontal = np.hypot(xyz[:, 0], xyz[:, 1])
        valid = np.isfinite(xyz).all(axis=1) & (horizontal > 1e-3)
        angles = np.arctan2(xyz[valid, 2], horizontal[valid])
        if angles.size:
            values.append(angles[:remaining]); remaining -= min(angles.size, remaining)
        if remaining <= 0: break
    if not values:
        raise ValueError("cannot estimate elevations: scans contain no finite non-zero points")
    samples = np.concatenate(values)
    if samples.size < beam_count * 8:
        raise ValueError(f"cannot estimate {beam_count} elevations from only {samples.size} valid points")
    centers = np.quantile(samples, (np.arange(beam_count) + 0.5) / beam_count)
    for _ in range(40):
        boundaries = (centers[:-1] + centers[1:]) / 2
        groups = np.searchsorted(boundaries, samples)
        counts = np.bincount(groups, minlength=beam_count)
        updated = np.divide(np.bincount(groups, weights=samples, minlength=beam_count), counts, out=centers.copy(), where=counts > 0)
        if np.max(np.abs(updated - centers)) < 1e-7: break
        centers = updated
    centers = np.sort(centers.astype(np.float32))
    if centers.shape != (beam_count,) or not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0) or np.max(np.abs(centers)) > np.pi / 2:
        raise ValueError("data-derived elevation estimate is invalid; supply a calibrated [64] radian table")
    return centers
