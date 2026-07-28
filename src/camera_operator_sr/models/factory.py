"""Explicit model construction for legacy and relation checkpoints."""

from torch import nn

from .relation.lidar_model import RelationLidarModel
from .relation.guided_model import CameraGuidedRelationModel
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
    if model_type == "relation_guided":
        l0_config = config.pop("l0_model_config")
        for key in ("base_model_type", "candidate_layout", "anchor_slots", "camera_token_layout", "camera_token_dim", "require_center_camera_valid", "depth_representation", "architecture_version", "horizontal_radius"):
            config.pop(key, None)
        return CameraGuidedRelationModel(RelationLidarModel(**{key: value for key, value in l0_config.items() if key not in {"model_type", "candidate_layout", "anchor_slots"}}), **config)
    raise ValueError(f"unsupported model_type: {model_type}")
