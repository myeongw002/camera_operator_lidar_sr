import torch
from torch import Tensor


def assert_shared_geometry(values: Tensor, name: str = "geometry") -> None:
    """Reject batches that would silently use only sample zero's geometry."""
    if values.ndim < 2:
        return
    reference = values[:1].expand_as(values)
    if not torch.allclose(values, reference, atol=1e-6, rtol=0.0):
        raise ValueError(f"All samples in a batch must share the same {name}.")


def validate_full_vertical_coverage(input_elevation: Tensor, target_elevation: Tensor, *, atol: float = 1e-5) -> None:
    """Fail fast unless input beams span every target beam physically.

    Row storage may be ascending or descending; all checks therefore use actual
    elevations rather than row indices.  The official L0 contract includes the
    lowest and highest target elevations in the 16 input beams.
    """
    if input_elevation.ndim != 1 or target_elevation.ndim != 1:
        raise ValueError("input and target elevation must be one-dimensional")
    if not torch.isfinite(input_elevation).all() or not torch.isfinite(target_elevation).all():
        raise ValueError("input and target elevation must be finite")
    if input_elevation.numel() < 2 or target_elevation.numel() < 2:
        raise ValueError("input and target elevation require at least two beams")
    if torch.unique(input_elevation).numel() != input_elevation.numel():
        raise ValueError("input elevation contains duplicate beams")
    for values, name in ((input_elevation, "input"), (target_elevation, "target")):
        delta = values.diff()
        if not (delta.gt(0).all() or delta.lt(0).all()):
            raise ValueError(f"{name} elevation must be strictly monotonic")
    input_min, input_max = input_elevation.min(), input_elevation.max()
    target_min, target_max = target_elevation.min(), target_elevation.max()
    if not torch.isclose(input_min, target_min, atol=atol, rtol=0.0) or not torch.isclose(input_max, target_max, atol=atol, rtol=0.0):
        raise ValueError("input elevation must include the target vertical FOV endpoints")
    if target_min < input_min - atol or target_max > input_max + atol:
        raise ValueError("target elevation lies outside the input vertical FOV")
