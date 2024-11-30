import torch
import torch.nn as nn

from network.blocks import DownResBlock, UpResBlock


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList(
            [
                DownResBlock(3, 64),
                DownResBlock(64, 128),
                DownResBlock(128, 256),
                DownResBlock(256, 512),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpResBlock(512, 256),
                UpResBlock(256, 128),
            ]
        )
        self.coral_head = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x)
            if i < len(self.down_blocks) - 1:
                skips.append(x)

        for up_block, skip in zip(self.up_blocks, skips[::-1]):
            x = up_block(x)
            x = x + skip

        return self.coral_head(x)
