"""L0 LiDAR-only local-relation correction model."""

import torch
from torch import Tensor, nn

from camera_operator_sr.geometry.candidate_graph import CandidateIndex, build_candidate_index
from camera_operator_sr.geometry.validation import assert_shared_geometry, validate_full_vertical_coverage

from .geometric_prior import VerticalLinearPrior
from .lidar_relation import LidarRelationMLP
from .local_context import build_local_candidate_context
from .outputs import RelationOutput
from .point_encoder import RelationPointEncoder
from .relation_aggregator import RelationAggregator


def corrected_two_anchor_weights(prior_weights: Tensor, anchor_valid: Tensor, correction: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Correct only two-valid-anchor logits; one/zero-anchor prior rules remain exact."""
    delta = correction.squeeze(1)
    valid = anchor_valid.bool()
    both = valid.all(dim=-1)
    logits = torch.log(prior_weights.clamp_min(eps))
    signs = torch.tensor([-1.0, 1.0], device=logits.device, dtype=logits.dtype)
    corrected = torch.softmax(logits + delta[..., None] * signs, dim=-1)
    return torch.where(both[..., None], corrected, prior_weights)


class RelationLidarModel(nn.Module):
    """L0: relation MLP corrections over B0's lower/upper center anchors."""

    model_type = "relation_l0"

    def __init__(self, horizontal_radius: int = 1, point_input_dim: int = 9, point_hidden_dim: int = 24, relation_hidden_dim: int = 64, correction_limit: float = 3.0, use_intensity: bool = True, architecture_version: int = 1):
        super().__init__()
        if horizontal_radius != 1:
            raise ValueError("L0 V1 requires horizontal_radius=1 for the fixed 2x3 candidate contract")
        if point_input_dim != 9:
            raise ValueError("L0 V1 candidate token dimension is fixed at 9")
        self.horizontal_radius = horizontal_radius
        self.model_config = {
            "model_type": self.model_type, "architecture_version": architecture_version,
            "horizontal_radius": horizontal_radius,
            "candidate_layout": "lower[-1,0,+1],upper[-1,0,+1]", "anchor_slots": [1, 4],
            "point_input_dim": point_input_dim, "point_hidden_dim": point_hidden_dim,
            "relation_hidden_dim": relation_hidden_dim, "correction_limit": correction_limit,
            "use_intensity": use_intensity,
        }
        self.use_intensity = use_intensity
        self.prior = VerticalLinearPrior()
        self.point_encoder = RelationPointEncoder(point_input_dim, point_hidden_dim)
        self.aggregator = RelationAggregator(point_hidden_dim)
        self.relation_head = LidarRelationMLP(self.aggregator.relation_dim, relation_hidden_dim, correction_limit)
        self._candidate_cache: dict[tuple, CandidateIndex] = {}

    def candidate_index(self, input_elevation: Tensor, target_elevation: Tensor, width: int) -> CandidateIndex:
        key = (str(input_elevation.device), tuple(input_elevation.detach().cpu().tolist()), tuple(target_elevation.detach().cpu().tolist()), width, self.horizontal_radius)
        if key not in self._candidate_cache:
            self._candidate_cache[key] = build_candidate_index(input_elevation, target_elevation, width, self.horizontal_radius)
        return self._candidate_cache[key]

    def build_state(self, batch: dict) -> dict:
        """Build the one-pass L0 state reused by the camera-guided wrapper."""
        lidar, target = batch["lidar"], batch["target"]
        for values, name in ((lidar["elevation"], "input elevation"), (target["elevation"], "target elevation"), (lidar["azimuth"], "azimuth")):
            if values.ndim == 2: assert_shared_geometry(values, name)
        input_elevation = lidar["elevation"][0] if lidar["elevation"].ndim == 2 else lidar["elevation"]
        target_elevation = target["elevation"][0] if target["elevation"].ndim == 2 else target["elevation"]
        validate_full_vertical_coverage(input_elevation, target_elevation)
        index = self.candidate_index(input_elevation, target_elevation, lidar["range"].shape[-1])
        intensity = lidar["intensity"] if self.use_intensity else torch.zeros_like(lidar["intensity"])
        context = build_local_candidate_context(lidar["range"], intensity, lidar["valid"], index)
        slots = (index.lower_center_slot, index.upper_center_slot)
        anchor_ranges, anchor_valid = context.ranges[..., slots], context.valid[..., slots]
        prior = self.prior(anchor_ranges, anchor_valid, index.query_fraction)
        embeddings = self.point_encoder(context.tokens, context.valid)
        relation_feature = self.aggregator(embeddings, context.valid, anchor_valid, prior.weights, index.query_fraction, context.normalized_log_range[..., slots])
        correction = self.relation_head(relation_feature)
        final_weights = corrected_two_anchor_weights(prior.weights, anchor_valid, correction)
        prediction = (final_weights * anchor_ranges).sum(dim=-1)
        prediction = torch.where(prior.has_anchor.squeeze(1), prediction, torch.zeros_like(prediction))[:, None]
        return {"index": index, "context": context, "prior": prior, "anchor_ranges": anchor_ranges,
                "anchor_valid": anchor_valid, "relation_feature": relation_feature, "correction": correction,
                "final_weights": final_weights, "prediction": prediction}

    def forward(self, batch: dict) -> RelationOutput:
        state = self.build_state(batch)
        return RelationOutput(state["prior"].weights, state["final_weights"], state["correction"],
                              state["anchor_ranges"], state["anchor_valid"], state["prior"].has_anchor,
                              state["prediction"], state["relation_feature"])
