import numpy as np
import torch

from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import LidarInferenceDataset
from camera_operator_sr.models.student import LidarOperatorStudent


def test_inference_dataset_requires_no_target_gt(tmp_path):
    frame = tmp_path / "00" / "000000"
    frame.mkdir(parents=True)
    for name in ("input_range.npy", "input_intensity.npy", "input_valid.npy"):
        np.save(frame / name, np.ones((4, 16), dtype=np.float32))
    np.savez(frame / "meta.npz", input_elevation=np.linspace(-0.2, 0.2, 4, dtype=np.float32), target_elevation=np.linspace(-0.2, 0.2, 8, dtype=np.float32), azimuth=np.linspace(-3.0, 3.0, 16, dtype=np.float32))
    sample = LidarInferenceDataset(tmp_path)[0]
    assert "range" not in sample["target"] and "camera" not in sample and "calibration" not in sample
    output = LidarOperatorStudent(lidar_feature_dim=16, hidden_dim=24)(collate_frames([sample]))
    assert output.predicted_range.shape == (1, 1, 8, 16)
    assert torch.isfinite(output.predicted_range).all()
