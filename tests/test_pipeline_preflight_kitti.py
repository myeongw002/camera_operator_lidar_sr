import numpy as np
import pytest

from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import synthetic_config, write_config


def test_kitti_validation_checks_elevation_and_common_stems(tmp_path):
    cfg = synthetic_config(tmp_path); cfg["dataset"].update(type="kitti_odometry", raw_root=str(tmp_path / "raw"), target_elevation_file=str(tmp_path / "elevation.npy"))
    cfg["stages"].update(prepare_range_images=True); np.save(tmp_path / "elevation.npy", np.linspace(-.4, .2, 64))
    root = tmp_path / "raw" / "sequences" / "00"; (root / "velodyne").mkdir(parents=True); (root / "image_2").mkdir(); (root / "velodyne" / "000000.bin").write_bytes(b"")
    (root / "calib.txt").write_text("P2: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    path = write_config(tmp_path, cfg); runner = PipelineRunner(cfg, path)
    with pytest.raises(ValueError, match="common"): runner._dataset_validation()


def test_split_kitti_roots_are_used_for_commands(tmp_path):
    cfg = synthetic_config(tmp_path); cfg["dataset"].update(type="kitti_odometry", scan_root=str(tmp_path / "velo"), image_root=str(tmp_path / "color"), calibration_root=str(tmp_path / "calib"), target_elevation_file=str(tmp_path / "elevation.npy"))
    np.save(tmp_path / "elevation.npy", np.linspace(-.4, .2, 64))
    path = write_config(tmp_path, cfg); runner = PipelineRunner(cfg, path)
    assert str(runner.context.scan_directory("00")) == str(tmp_path / "velo" / "sequences" / "00" / "velodyne")
    assert str(runner.context.image_directory("00")) == str(tmp_path / "color" / "sequences" / "00" / "image_2")
    assert str(runner.context.calibration_file("00")) == str(tmp_path / "calib" / "sequences" / "00" / "calib.txt")
