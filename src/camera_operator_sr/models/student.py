import torch
import torch.nn as nn
from torch import Tensor

from camera_operator_sr.geometry.candidate_graph import CandidateIndex, build_candidate_index
from camera_operator_sr.geometry.validation import assert_shared_geometry

from .lidar_encoder import LidarEncoder
from .operator_decoder import OperatorDecoder
from .outputs import OperatorOutput


def build_lidar_channels(batch: dict) -> Tensor:
    lidar = batch["lidar"]
    ranges, intensity, valid = lidar["range"], lidar["intensity"], lidar["valid"]
    elevation = lidar["elevation"]
    if elevation.ndim == 1:
        elevation = elevation[None].expand(ranges.shape[0], -1)
    phi = elevation[:, None, :, None].expand(-1, 1, -1, ranges.shape[-1])
    return torch.cat((torch.log(ranges.clamp_min(1e-3)) * valid, intensity, valid, torch.sin(phi), torch.cos(phi)), dim=1)


class LidarOperatorStudent(nn.Module):
    def __init__(self, lidar_feature_dim: int = 96, hidden_dim: int = 128, horizontal_radius: int = 1, residual_scale_alpha: float = 1.5, residual_min: float = 0.1, residual_max: float = 3.0):
        super().__init__()
        self.model_config = {"lidar_feature_dim": lidar_feature_dim, "hidden_dim": hidden_dim, "horizontal_radius": horizontal_radius, "residual_scale_alpha": residual_scale_alpha, "residual_min": residual_min, "residual_max": residual_max}
        self.encoder = LidarEncoder(5, lidar_feature_dim)
        self.decoder = OperatorDecoder(lidar_feature_dim, hidden_dim=hidden_dim, residual_scale_alpha=residual_scale_alpha, residual_min=residual_min, residual_max=residual_max)
        self.horizontal_radius = horizontal_radius
        self._candidate_cache: dict[tuple, CandidateIndex] = {}

    def candidate_index(self, input_elevation: Tensor, target_elevation: Tensor, width: int) -> CandidateIndex:
        # The cache is deliberately keyed by exact geometry, avoiding accidental checkpoint/data mismatch.
        key = (str(input_elevation.device), tuple(input_elevation.detach().cpu().tolist()), tuple(target_elevation.detach().cpu().tolist()), width, self.horizontal_radius)
        if key not in self._candidate_cache:
            self._candidate_cache[key] = build_candidate_index(input_elevation, target_elevation, width, self.horizontal_radius)
        return self._candidate_cache[key]

    def forward(self, batch: dict) -> OperatorOutput:
        lidar = batch["lidar"]
        if lidar["elevation"].ndim == 2:
            assert_shared_geometry(lidar["elevation"], "input elevation")
        if batch["target"]["elevation"].ndim == 2:
            assert_shared_geometry(batch["target"]["elevation"], "target elevation")
        if lidar["azimuth"].ndim == 2:
            assert_shared_geometry(lidar["azimuth"], "azimuth")
        input_elevation = lidar["elevation"][0] if lidar["elevation"].ndim == 2 else lidar["elevation"]
        target_elevation = batch["target"]["elevation"][0] if batch["target"]["elevation"].ndim == 2 else batch["target"]["elevation"]
        index = self.candidate_index(input_elevation, target_elevation, lidar["range"].shape[-1])
        return self.decoder(self.encoder(build_lidar_channels(batch)), lidar["range"], lidar["valid"], index)
