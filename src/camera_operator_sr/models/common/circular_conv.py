import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CircularConv2d(nn.Module):
    """Convolution with circular horizontal and replicate vertical padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int] = 3, stride: int | tuple[int, int] = 1, bias: bool = False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.pad_h, self.pad_w = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.pad_w:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        if self.pad_h:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="replicate")
        return self.conv(x)
