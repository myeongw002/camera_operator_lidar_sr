import numpy as np

from camera_operator_sr.pipeline.validation import validate_depth_frame


def test_float32_depth_preserves_large_finite_inverse_depth(tmp_path):
    depth = np.array([[100_000.0]], dtype=np.float32)
    valid = np.array([[1]], dtype=np.uint8)
    np.save(tmp_path / "relative_depth.npy", depth)
    np.save(tmp_path / "depth_valid.npy", valid)
    validate_depth_frame(tmp_path)
    assert np.isfinite(np.load(tmp_path / "relative_depth.npy")).all()
