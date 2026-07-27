"""Outputs shared by the incremental relation-model implementations."""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class RelationOutput:
    """Two-anchor relation prediction.

    Anchor tensors use ``[lower, upper]`` order and have shape ``[B,H,W,2]``.
    Image-shaped tensors use ``[B,1,H,W]``.  B0 has no learned relation
    feature, so ``relation_feature`` is ``None``.
    """

    prior_weights: Tensor
    final_weights: Tensor
    correction: Tensor
    anchor_ranges: Tensor
    anchor_valid: Tensor
    has_anchor: Tensor
    predicted_range: Tensor
    relation_feature: Tensor | None
