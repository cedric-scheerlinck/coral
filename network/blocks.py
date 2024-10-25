import torch
import torch.nn as nn


def conv_bn(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            conv_bn(in_channels, out_channels),
            nn.ReLU(inplace=True),
            conv_bn(out_channels, out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

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
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
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
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
