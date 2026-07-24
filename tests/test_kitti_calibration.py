import numpy as np

from camera_operator_sr.data.kitti_calibration import load_kitti_calibration


def test_kitti_calibration_composes_rectification_and_camera_offset(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text("P2: 10 0 5 20 0 10 4 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 1 0 0 1 0 1 0 2 0 0 1 3\nS_02: 1242 375\n")
    calibration = load_kitti_calibration(path)
    assert calibration.image_size == (375, 1242)
    assert np.allclose(calibration.K, [[10, 0, 5], [0, 10, 4], [0, 0, 1]])
    assert np.allclose(calibration.T_cam_lidar[:3, 3], [3, 2, 3])


def test_kitti_odometry_calibration_accepts_missing_rectification_and_tr_key(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text("P2: 10 0 5 20 0 10 4 0 0 0 1 0\nTr: 1 0 0 1 0 1 0 2 0 0 1 3\n")
    calibration = load_kitti_calibration(path)
    assert np.allclose(calibration.T_cam_lidar[:3, 3], [3, 2, 3])
