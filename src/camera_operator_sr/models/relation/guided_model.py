"""Training/evaluation-only camera-guided relation model G."""
import torch
from torch import Tensor, nn

from camera_operator_sr.geometry.candidate_projection import candidate_lidar_points
from camera_operator_sr.geometry.feature_sampling import sample_image_features
from camera_operator_sr.geometry.projection import project_lidar_points

from .camera_adapter import CameraRelationAdapter
from .lidar_model import RelationLidarModel, corrected_two_anchor_weights
from .local_context import robust_normalize_candidates
from .outputs import GuidedRelationOutput, RelationOutput


class CameraGuidedRelationModel(nn.Module):
    model_type = "relation_guided"
    def __init__(self, l0_model: RelationLidarModel, camera_point_hidden_dim: int = 16,
                 camera_relation_hidden_dim: int = 32, camera_correction_limit: float = 3.0):
        super().__init__()
        self.l0 = l0_model
        for parameter in self.l0.parameters(): parameter.requires_grad_(False)
        self.l0.eval()
        self.horizontal_radius = l0_model.horizontal_radius
        self.camera_adapter = CameraRelationAdapter(l0_model.aggregator.relation_dim, 9, camera_point_hidden_dim, camera_relation_hidden_dim, camera_correction_limit)
        self.model_config = {"model_type": self.model_type, "architecture_version": 1, "base_model_type": "relation_l0", "l0_model_config": dict(l0_model.model_config), "candidate_layout": "lower[-1,0,+1],upper[-1,0,+1]", "anchor_slots": [1, 4], "horizontal_radius": self.horizontal_radius, "camera_token_layout": "normalized_relative_inverse_depth,normalized_log_range,delta_elevation,sin_delta_azimuth,cos_delta_azimuth,camera_valid,lidar_valid,upper_side,horizontal_offset", "camera_token_dim": 9, "camera_point_hidden_dim": camera_point_hidden_dim, "camera_relation_hidden_dim": camera_relation_hidden_dim, "camera_correction_limit": camera_correction_limit, "require_center_camera_valid": True, "depth_representation": "relative_inverse_depth"}

    def train(self, mode: bool = True):
        super().train(mode); self.l0.eval(); return self

    def _camera_samples(self, batch: dict, state: dict) -> tuple[Tensor, Tensor, Tensor]:
        context, index = state["context"], state["index"]
        xyz, lidar_valid = candidate_lidar_points(context.ranges, context.valid, index, batch["lidar"]["elevation"], batch["lidar"]["azimuth"])
        b, h, w, k, _ = xyz.shape
        depth = batch["camera"]["relative_depth"]
        projection = project_lidar_points(xyz.reshape(b, -1, 3), batch["calibration"]["K"], batch["calibration"]["T_cam_lidar"], depth.shape[-2:])
        sampled, _ = sample_image_features(depth, projection.normalized_grid, projection.valid)
        sampled_valid, _ = sample_image_features(batch["camera"]["depth_valid"].to(depth.dtype), projection.normalized_grid, projection.valid)
        sampled = sampled.reshape(b, h, w, k)
        valid = (lidar_valid & projection.valid.reshape(b, h, w, k)
                 & sampled_valid.reshape(b, h, w, k).gt(.5)
                 & torch.isfinite(sampled))
        return torch.where(valid, sampled, torch.zeros_like(sampled)), valid, lidar_valid

    def forward(self, batch: dict) -> GuidedRelationOutput:
        with torch.no_grad(): state = self.l0.build_state(batch)
        lidar = RelationOutput(state["prior"].weights, state["final_weights"], state["correction"], state["anchor_ranges"], state["anchor_valid"], state["prior"].has_anchor, state["prediction"], state["relation_feature"])
        camera_depth, camera_valid, lidar_valid = self._camera_samples(batch, state)
        # relative_depth.npy is relative inverse depth, not metric depth and
        # must therefore be normalized directly rather than log-transformed.
        normalized_depth = robust_normalize_candidates(camera_depth, camera_valid)
        context, index = state["context"], state["index"]
        b, h, w, k = camera_depth.shape
        delta_elev = index.delta_elevation.to(camera_depth.device, camera_depth.dtype)[None, :, None, :].expand(b, -1, w, -1)
        delta_az = index.delta_azimuth.to(camera_depth.device, camera_depth.dtype)[None, None, None, :].expand(b, h, w, -1)
        slots = torch.arange(k, device=camera_depth.device)
        upper = slots.ge(k // 2).to(camera_depth.dtype)[None, None, None, :].expand(b, h, w, -1)
        horizontal = index.column_offsets.to(camera_depth.device, camera_depth.dtype)[None, None, None, :].expand(b, h, w, -1)
        tokens = torch.stack((normalized_depth, context.normalized_log_range, delta_elev, torch.sin(delta_az), torch.cos(delta_az), camera_valid.to(camera_depth.dtype), lidar_valid.to(camera_depth.dtype), upper, horizontal), dim=-1) * camera_valid[..., None].to(camera_depth.dtype)
        raw_camera, feature = self.camera_adapter(tokens, camera_valid, state["relation_feature"])
        guidance = state["anchor_valid"].bool().all(-1) & camera_valid[..., index.lower_center_slot] & camera_valid[..., index.upper_center_slot]
        camera_correction = torch.where(guidance[:, None], raw_camera, torch.zeros_like(raw_camera))
        total = lidar.correction + camera_correction
        weights = corrected_two_anchor_weights(lidar.prior_weights, lidar.anchor_valid, total)
        prediction = (weights * lidar.anchor_ranges).sum(-1)
        prediction = torch.where(lidar.has_anchor.squeeze(1), prediction, torch.zeros_like(prediction))[:, None]
        return GuidedRelationOutput(lidar, camera_correction, total, weights, prediction, camera_depth, camera_valid, guidance[:, None], feature)
