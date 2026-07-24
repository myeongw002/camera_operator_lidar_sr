import numpy as np
import pytest

from camera_operator_sr.data.dataset import ProcessedTrainingDataset


def _frame(root, sequence, frame):
    path = root / sequence / f"{frame:06d}"; path.mkdir(parents=True)
    for name in ("target_range.npy", "target_valid.npy", "input_range.npy", "input_intensity.npy", "input_valid.npy", "relative_depth.npy", "depth_valid.npy"): np.save(path / name, np.ones((1, 2), dtype=np.float32))
    np.savez(path / "meta.npz", input_elevation=np.array([0], dtype=np.float32), target_elevation=np.array([0], dtype=np.float32), azimuth=np.array([0, 1], dtype=np.float32), K=np.eye(3), T_cam_lidar=np.eye(4))


def test_frame_shuffle_is_seeded_and_prefers_other_sequence(tmp_path):
    for sequence in ("00", "01"):
        for frame in (0, 200): _frame(tmp_path, sequence, frame)
    one = ProcessedTrainingDataset(tmp_path, depth_mode="frame_shuffled", shuffle_seed=7)
    two = ProcessedTrainingDataset(tmp_path, depth_mode="frame_shuffled", shuffle_seed=7)
    assert one._frame_shuffle == two._frame_shuffle
    assert all(one.frames[source].parent.name != frame.parent.name for frame, source in zip(one.frames, one._frame_shuffle))
    with pytest.raises(ValueError): ProcessedTrainingDataset(tmp_path / "00", frame_dirs=[tmp_path / "00" / "000000"], depth_mode="frame_shuffled")
