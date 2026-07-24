import numpy as np
import pytest

from camera_operator_sr.pipeline.validation import validate_range_frame


def test_range_validation_detects_deleted_required_output(tmp_path):
    frame = tmp_path / "00" / "000000"; frame.mkdir(parents=True)
    for name, shape in (("input_range.npy", (2, 4)), ("input_intensity.npy", (2, 4)), ("input_valid.npy", (2, 4)), ("target_range.npy", (3, 4)), ("target_valid.npy", (3, 4))): np.save(frame / name, np.ones(shape, np.float32))
    np.savez(frame / "meta.npz", input_elevation=np.ones(2), target_elevation=np.ones(3), azimuth=np.ones(4))
    validate_range_frame(frame); (frame / "input_range.npy").unlink()
    with pytest.raises(FileNotFoundError): validate_range_frame(frame)
