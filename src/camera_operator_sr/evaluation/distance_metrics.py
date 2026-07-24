"""GT-range distance bin parsing and masks for count-weighted evaluation."""

import math

import torch
from torch import Tensor


DEFAULT_DISTANCE_BINS: tuple[float, ...] = (0.0, 10.0, 20.0, 40.0, 60.0, 80.0, math.inf)


def parse_distance_bins(values: list[str] | tuple[str, ...] | None) -> tuple[float, ...]:
    raw = [str(value).strip().lower() for value in values] if values is not None else [str(value) for value in DEFAULT_DISTANCE_BINS]
    if len(raw) < 2:
        raise ValueError("Distance bins require at least two boundaries.")
    try:
        bins = tuple(float("inf") if value in {"inf", "+inf", "infinity", "+infinity"} else float(value) for value in raw)
    except ValueError as exc:
        raise ValueError("Distance bins must be numeric values or 'inf'.") from exc
    if not math.isfinite(bins[0]) or bins[0] < 0:
        raise ValueError("The first distance bin boundary must be finite and non-negative.")
    if any(not right > left for left, right in zip(bins, bins[1:])):
        raise ValueError("Distance bins must be strictly increasing with no duplicates.")
    if any(math.isinf(value) for value in bins[:-1]):
        raise ValueError("Only the final distance bin boundary may be inf.")
    return bins


def gt_distance_bin_mask(target_range: Tensor, target_valid: Tensor, minimum: float, maximum: float) -> Tensor:
    """Select valid GT targets in [minimum, maximum), never predicted ranges."""
    if not maximum > minimum:
        raise ValueError("Distance bin maximum must exceed minimum.")
    return target_valid.bool() & target_range.ge(minimum) & target_range.lt(maximum)


def serialize_distance_boundary(value: float) -> str | float:
    return "inf" if math.isinf(value) else value
