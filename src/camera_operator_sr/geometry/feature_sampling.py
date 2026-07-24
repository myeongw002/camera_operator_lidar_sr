import torch.nn.functional as F
from torch import Tensor


def sample_image_features(feature_map: Tensor, normalized_grid: Tensor, projection_valid: Tensor, align_corners: bool = False) -> tuple[Tensor, Tensor]:
    """Bilinearly sample features at [B,N,2] grids, zeroing invalid samples."""
    sampled = F.grid_sample(feature_map, normalized_grid[:, :, None, :], mode="bilinear", padding_mode="zeros", align_corners=align_corners)
    features = sampled.squeeze(-1).transpose(1, 2)
    confidence = projection_valid.to(feature_map.dtype)
    return features * confidence[..., None], confidence
