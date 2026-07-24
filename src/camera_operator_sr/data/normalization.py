import torch
from torch import Tensor


def robust_normalize_inverse_depth(depth: Tensor, valid: Tensor, clip_value: float = 5.0, eps: float = 1e-6) -> Tensor:
    output = torch.zeros_like(depth)
    for index in range(depth.shape[0]):
        values = depth[index][valid[index].bool()]
        if values.numel():
            median = values.median()
            mad = (values - median).abs().median()
            output[index] = ((depth[index] - median) / (mad + eps)).clamp(-clip_value, clip_value)
    return output * valid.to(depth.dtype)


def depth_gradients(depth: Tensor, valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
    gradient_x = torch.zeros_like(depth)
    gradient_y = torch.zeros_like(depth)
    gradient_x[..., :, :-1] = depth[..., :, 1:] - depth[..., :, :-1]
    gradient_y[..., :-1, :] = depth[..., 1:, :] - depth[..., :-1, :]
    if valid is not None:
        valid = valid.bool()
        horizontal = torch.zeros_like(valid)
        vertical = torch.zeros_like(valid)
        horizontal[..., :, :-1] = valid[..., :, :-1] & valid[..., :, 1:]
        vertical[..., :-1, :] = valid[..., :-1, :] & valid[..., 1:, :]
        gradient_x *= horizontal.to(depth.dtype)
        gradient_y *= vertical.to(depth.dtype)
    return gradient_x, gradient_y


def build_depth_channels(depth: Tensor, valid: Tensor) -> Tensor:
    normalized = robust_normalize_inverse_depth(depth, valid)
    gradient_x, gradient_y = depth_gradients(normalized, valid)
    return torch.cat((normalized, gradient_x, gradient_y, valid.to(depth.dtype)), dim=1)
