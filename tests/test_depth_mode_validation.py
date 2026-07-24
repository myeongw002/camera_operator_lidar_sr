import pytest

from camera_operator_sr.training.checkpoint import validate_teacher_depth_mode


def test_teacher_depth_mode_requires_match_or_explicit_warning():
    checkpoint={"depth_mode":"correct"}
    validate_teacher_depth_mode(checkpoint,"correct")
    with pytest.raises(ValueError,match="depth mode mismatch"): validate_teacher_depth_mode(checkpoint,"none")
    with pytest.warns(UserWarning,match="explicitly allowed"): validate_teacher_depth_mode(checkpoint,"none",allow_mismatch=True)
    with pytest.raises(ValueError,match="does not contain depth_mode"): validate_teacher_depth_mode({},"correct")
