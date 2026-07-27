import torch

from camera_operator_sr.losses.relation import relation_supervised_loss
from camera_operator_sr.models.relation import RelationLidarModel
from test_relation_lidar_model import batch


def test_relation_loss_masks_observed_and_returns_finite_empty_values():
    value = batch(); output = RelationLidarModel()(value); losses = relation_supervised_loss(output, value)
    assert losses["supported_count"].item() == 4
    assert losses["both_anchor_count"].item() == 4
    assert torch.isfinite(losses["loss"])
    value["lidar"]["valid"].zero_(); losses = relation_supervised_loss(RelationLidarModel()(value), value)
    assert losses["supported_count"].item() == 0 and losses["loss"].item() == 0
