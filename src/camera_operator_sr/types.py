from dataclasses import dataclass

from torch import Tensor


@dataclass
class RangeImage:
    range: Tensor
    intensity: Tensor
    valid: Tensor


@dataclass
class FrameSample:
    lidar_range: Tensor
    lidar_intensity: Tensor
    lidar_valid: Tensor
    target_range: Tensor
    target_valid: Tensor
    relative_depth: Tensor
    depth_valid: Tensor
    input_elevation: Tensor
    target_elevation: Tensor
    azimuth: Tensor
    camera_intrinsic: Tensor
    camera_from_lidar: Tensor
