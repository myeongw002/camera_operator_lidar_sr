import numpy as np

from camera_operator_sr.pipeline.elevation import estimate_elevations
from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import synthetic_config, write_config


def _scan(path):
    angles = np.linspace(-0.42, 0.04, 64, dtype=np.float32)
    azimuth = np.linspace(-3.0, 3.0, 32, dtype=np.float32)
    rows = []
    for elevation in angles:
        for heading in azimuth:
            rows.append([10 * np.cos(elevation) * np.cos(heading), 10 * np.cos(elevation) * np.sin(heading), 10 * np.sin(elevation), 1])
    np.asarray(rows, np.float32).tofile(path)


def test_elevation_estimation_uses_real_scan_vertical_angles(tmp_path):
    scan = tmp_path / "000000.bin"; _scan(scan)
    estimated = estimate_elevations([scan])
    assert estimated.shape == (64,) and np.all(np.diff(estimated) > 0)
    assert np.allclose(estimated, np.linspace(-0.42, 0.04, 64), atol=1e-4)


def test_pipeline_generates_missing_elevation_table_from_scans(tmp_path):
    cfg = synthetic_config(tmp_path); cfg["dataset"].update(type="kitti_odometry", scan_root=str(tmp_path / "velo"), image_root=str(tmp_path / "color"), calibration_root=str(tmp_path / "calib"), target_elevation_file=str(tmp_path / "missing.npy"), auto_estimate_elevation=True)
    for sequence in ("00",):
        scan_dir = tmp_path / "velo" / "sequences" / sequence / "velodyne"; image_dir = tmp_path / "color" / "sequences" / sequence / "image_2"; calibration = tmp_path / "calib" / "sequences" / sequence
        scan_dir.mkdir(parents=True); image_dir.mkdir(parents=True); calibration.mkdir(parents=True)
        _scan(scan_dir / "000000.bin"); (image_dir / "000000.png").write_bytes(b"placeholder")
        (calibration / "calib.txt").write_text("P2: 1 0 0 0 0 1 0 0 0 0 1 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    path = write_config(tmp_path, cfg); runner = PipelineRunner(cfg, path); runner._dataset_validation()
    table = runner.context.root / "calibration" / "estimated_hdl64_elevations.npy"
    assert table.exists() and np.load(table).shape == (64,)
    assert runner.context.target_elevation_file == table
