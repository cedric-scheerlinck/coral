import torch
import torch.nn as nn

USE_BATCH_NORM = True


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = True,
) -> nn.Sequential:
    padding = get_padding(kernel_size, stride)
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    if USE_BATCH_NORM:
        return nn.Sequential(conv, nn.BatchNorm2d(out_channels))
    return conv


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            conv_bn(in_channels, out_channels),
            nn.ReLU(inplace=True),
            conv_bn(out_channels, out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = conv_bn(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv_block(x) + self.shortcut(x))


class DownResBlock(ResBlock):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.conv_block = nn.Sequential(
            conv_bn(in_channels, out_channels, stride=2),
            nn.ReLU(inplace=True),
            conv_bn(out_channels, out_channels),
        )
        self.shortcut = conv_bn(
            in_channels, out_channels, kernel_size=1, stride=2, bias=False
        )


class UpResBlock(ResBlock):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.conv_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_bn(in_channels, out_channels),
            nn.ReLU(inplace=True),
            conv_bn(out_channels, out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_bn(in_channels, out_channels, kernel_size=1, bias=False),
        )
