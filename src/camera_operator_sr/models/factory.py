"""Explicit model construction for legacy and relation checkpoints."""

from torch import nn

from .relation.lidar_model import RelationLidarModel
from .student import LidarOperatorStudent


def build_model(model_config: dict) -> nn.Module:
    config = dict(model_config)
    model_type = config.pop("model_type", "legacy_operator")
    if model_type == "legacy_operator":
        config.pop("architecture_version", None)
        return LidarOperatorStudent(**config)
    if model_type == "relation_l0":
        config.pop("candidate_layout", None)
        config.pop("anchor_slots", None)
        return RelationLidarModel(**config)
    raise ValueError(f"unsupported model_type: {model_type}")
