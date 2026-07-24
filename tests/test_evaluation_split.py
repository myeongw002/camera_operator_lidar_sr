import numpy as np

from camera_operator_sr.data.dataset import ProcessedTrainingDataset


def test_split_selects_only_explicit_test_frame(tmp_path):
    for sequence, frame in (("00", "000000"), ("09", "000001")):
        path = tmp_path / sequence / frame; path.mkdir(parents=True)
        for name in ("target_range.npy", "target_valid.npy", "input_range.npy", "input_intensity.npy", "input_valid.npy"): np.save(path / name, np.ones((1, 2), dtype=np.float32))
        np.savez(path / "meta.npz", input_elevation=np.array([0],dtype=np.float32), target_elevation=np.array([0],dtype=np.float32), azimuth=np.array([0,1],dtype=np.float32), K=np.eye(3), T_cam_lidar=np.eye(4))
    split=tmp_path/"test.txt"; split.write_text("09/000001\n")
    dataset=ProcessedTrainingDataset(tmp_path,split_file=split)
    assert len(dataset)==1 and dataset.frames[0].parent.name=="09"
