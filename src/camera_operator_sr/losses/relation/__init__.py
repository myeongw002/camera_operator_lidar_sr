"""Losses for relation models."""

from .supervised import relation_supervised_loss
from .guidance import guided_relation_supervised_loss

__all__ = ["guided_relation_supervised_loss", "relation_supervised_loss"]
