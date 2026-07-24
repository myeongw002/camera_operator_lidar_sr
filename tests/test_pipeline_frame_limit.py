import numpy as np

from camera_operator_sr.pipeline.runner import PipelineRunner
from pipeline_support import synthetic_config, write_config


def test_frame_manifest_limits_processed_frames_and_splits(tmp_path):
    cfg = synthetic_config(tmp_path); cfg["dataset"]["type"] = "processed_synthetic"; cfg["dataset"]["max_frames_per_sequence"] = 3
    root = tmp_path / "processed" / "00"
    for index in range(10):
        frame = root / f"{index:06d}"; frame.mkdir(parents=True); np.savez(frame / "meta.npz", x=np.array([1]))
        np.save(frame / "target_range.npy", np.ones((3, 2), np.float32))
    path = write_config(tmp_path, cfg); runner = PipelineRunner(cfg, path); runner._dataset_validation(); runner._create_splits()
    names = (runner.context.frame_manifest("00")).read_text().splitlines()
    assert names == ["000000", "000001", "000002"]
    assert runner.context.train_split.read_text().splitlines() == [f"00/{name}" for name in names]
