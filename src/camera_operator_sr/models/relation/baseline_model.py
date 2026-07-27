"""B0: fixed geometric prior with no trainable parameters or checkpoint."""

import torch
from torch import Tensor, nn

from camera_operator_sr.geometry.candidate_graph import CandidateIndex, build_candidate_index, gather_candidate_values
from camera_operator_sr.geometry.validation import assert_shared_geometry

from .geometric_prior import VerticalLinearPrior
from .outputs import RelationOutput


class GeometricBaselineModel(nn.Module):
    """Interpolate 16-channel LiDAR to target rows using physical elevation.

    The 2x3 candidate graph is gathered for the shared relation contract, but
    B0 interpolates only the lower-center and upper-center slots.  Exact input
    rows naturally retain their observed range through the one-anchor rule.
    """

    model_type = "geometric_prior"

    def __init__(self, horizontal_radius: int = 1):
        super().__init__()
        if horizontal_radius < 0:
            raise ValueError("horizontal_radius must be non-negative")
        self.horizontal_radius = horizontal_radius
        self.prior = VerticalLinearPrior()
        self._candidate_cache: dict[tuple, CandidateIndex] = {}

    def candidate_index(self, input_elevation: Tensor, target_elevation: Tensor, width: int) -> CandidateIndex:
        key = (
            str(input_elevation.device), tuple(input_elevation.detach().cpu().tolist()),
            tuple(target_elevation.detach().cpu().tolist()), width, self.horizontal_radius,
        )
        if key not in self._candidate_cache:
            self._candidate_cache[key] = build_candidate_index(input_elevation, target_elevation, width, self.horizontal_radius)
        return self._candidate_cache[key]

    def forward(self, batch: dict) -> RelationOutput:
        lidar, target = batch["lidar"], batch["target"]
        self._assert_shared_batch_geometry(lidar, target)
        input_elevation = lidar["elevation"][0] if lidar["elevation"].ndim == 2 else lidar["elevation"]
        target_elevation = target["elevation"][0] if target["elevation"].ndim == 2 else target["elevation"]
        index = self.candidate_index(input_elevation, target_elevation, lidar["range"].shape[-1])
        candidate_ranges = gather_candidate_values(lidar["range"], index).squeeze(-1)
        observed_valid = gather_candidate_values(lidar["valid"], index).squeeze(-1).bool()
        geometric_valid = index.geometric_valid.to(lidar["valid"].device)[None, :, None, :]
        candidate_valid = observed_valid & geometric_valid
        anchor_slots = (index.lower_center_slot, index.upper_center_slot)
        anchors = candidate_ranges[..., anchor_slots]
        anchor_valid = candidate_valid[..., anchor_slots]
        prior = self.prior(anchors, anchor_valid, index.query_fraction)
        correction = torch.zeros_like(prior.predicted_range)
        return RelationOutput(
            prior_weights=prior.weights,
            final_weights=prior.weights,
            correction=correction,
            anchor_ranges=prior.anchor_ranges,
            anchor_valid=prior.anchor_valid,
            has_anchor=prior.has_anchor,
            predicted_range=prior.predicted_range,
            relation_feature=None,
        )

    @staticmethod
    def _assert_shared_batch_geometry(lidar: dict, target: dict) -> None:
        for values, name in ((lidar["elevation"], "input elevation"), (target["elevation"], "target elevation"), (lidar["azimuth"], "azimuth")):
            if values.ndim == 2:
                assert_shared_geometry(values, name)
