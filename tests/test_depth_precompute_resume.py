import numpy as np
import pytest

from camera_operator_sr.pipeline.validation import validate_depth_frame


def test_depth_validation_accepts_complete_frame_and_rejects_partial(tmp_path):
    np.save(tmp_path / "relative_depth.npy", np.ones((2, 3), np.float32)); np.save(tmp_path / "depth_valid.npy", np.ones((2, 3), np.uint8))
    validate_depth_frame(tmp_path); (tmp_path / "depth_valid.npy").unlink()
    with pytest.raises(FileNotFoundError): validate_depth_frame(tmp_path)
