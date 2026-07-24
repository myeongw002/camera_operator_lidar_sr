import numpy as np
import pytest

from camera_operator_sr.pipeline.validation import validate_inference


def test_inference_validation_checks_arrays_and_finiteness(tmp_path):
    path = tmp_path / "x.npz"; values = {"range": np.ones((1, 1, 2, 3)), "return_probability": np.ones((1, 1, 2, 3)), "anchor_entropy": np.ones((1, 2, 3)), "residual": np.ones((1, 1, 2, 3))}
    np.savez(path, **values); validate_inference(path); values["range"][0, 0, 0, 0] = np.nan; np.savez(path, **values)
    with pytest.raises(ValueError): validate_inference(path)
