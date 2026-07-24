import torch
from torch import Tensor


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Softmax with all-invalid slices represented by all-zero weights."""
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    has_valid = mask.any(dim=dim, keepdim=True)
    safe_logits = torch.where(has_valid, masked_logits, torch.zeros_like(masked_logits))
    return torch.softmax(safe_logits, dim=dim) * mask.to(logits.dtype)
