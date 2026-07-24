import torch
from torch import Tensor


def assert_shared_geometry(values: Tensor, name: str = "geometry") -> None:
    """Reject batches that would silently use only sample zero's geometry."""
    if values.ndim < 2:
        return
    reference = values[:1].expand_as(values)
    if not torch.allclose(values, reference, atol=1e-6, rtol=0.0):
        raise ValueError(f"All samples in a batch must share the same {name}.")
