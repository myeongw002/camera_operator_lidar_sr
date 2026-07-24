import torch
from torch import Tensor

from camera_operator_sr.data.normalization import build_depth_channels
from camera_operator_sr.data.range_image import range_image_to_pointcloud
from camera_operator_sr.geometry.feature_sampling import sample_image_features
from camera_operator_sr.geometry.projection import project_lidar_points
from camera_operator_sr.geometry.validation import assert_shared_geometry

from .depth_encoder import DepthEncoder
from .fusion import ObservedAnchorFusion
from .student import LidarOperatorStudent, build_lidar_channels


class CameraGuidedOperatorTeacher(LidarOperatorStudent):
    """Teacher only samples camera features on physically observed input points."""
    def __init__(self, lidar_feature_dim: int = 96, depth_feature_dim: int = 32, **kwargs):
        super().__init__(lidar_feature_dim=lidar_feature_dim, **kwargs)
        self.model_config = self.model_config | {"depth_feature_dim": depth_feature_dim}
        self.depth_encoder = DepthEncoder(depth_feature_dim)
        self.fusion = ObservedAnchorFusion(lidar_feature_dim, depth_feature_dim)

    def forward(self, batch: dict):
        lidar = batch["lidar"]
        camera = batch["camera"]
        calibration = batch["calibration"]
        if lidar["elevation"].ndim == 2:
            assert_shared_geometry(lidar["elevation"], "input elevation")
        if batch["target"]["elevation"].ndim == 2:
            assert_shared_geometry(batch["target"]["elevation"], "target elevation")
        if lidar["azimuth"].ndim == 2:
            assert_shared_geometry(lidar["azimuth"], "azimuth")
        lidar_features = self.encoder(build_lidar_channels(batch))
        depth_features = self.depth_encoder(build_depth_channels(camera["relative_depth"], camera["depth_valid"]))
        points = range_image_to_pointcloud(lidar["range"].squeeze(1), lidar["valid"].squeeze(1), lidar["elevation"][0] if lidar["elevation"].ndim == 2 else lidar["elevation"], lidar["azimuth"][0] if lidar["azimuth"].ndim == 2 else lidar["azimuth"])
        b, h, w, _ = points.shape
        projection = project_lidar_points(points.reshape(b, h * w, 3), calibration["K"], calibration["T_cam_lidar"], camera["relative_depth"].shape[-2:])
        sampled, projection_confidence = sample_image_features(depth_features, projection.normalized_grid, projection.valid)
        # Camera confidence requires both a valid projection and a valid source depth sample.
        depth_valid_sample, _ = sample_image_features(camera["depth_valid"].to(depth_features.dtype), projection.normalized_grid, projection.valid)
        confidence = (projection_confidence * depth_valid_sample.squeeze(-1).gt(0.5)).reshape(b, 1, h, w).to(lidar_features.dtype)
        sampled = sampled.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        fused = self.fusion(lidar_features, sampled, confidence)
        input_elevation = lidar["elevation"][0] if lidar["elevation"].ndim == 2 else lidar["elevation"]
        target_elevation = batch["target"]["elevation"][0] if batch["target"]["elevation"].ndim == 2 else batch["target"]["elevation"]
        return self.decoder(fused, lidar["range"], lidar["valid"], self.candidate_index(input_elevation, target_elevation, w))
