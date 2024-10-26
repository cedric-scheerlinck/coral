import pytest
import torch
from network.blocks import DownResBlock
from network.blocks import ResBlock
from network.blocks import UpResBlock
from network.blocks import conv_bn

# Global constants
BATCH_SIZE = 2
IN_CHANNELS = 8
OUT_CHANNELS = 16
HEIGHT = 32
WIDTH = 32


@pytest.fixture
def input_tensor() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)


def test_conv_bn(input_tensor: torch.Tensor) -> None:
    conv_bn_layer = conv_bn(IN_CHANNELS, OUT_CHANNELS)
    out = conv_bn_layer(input_tensor)
    assert out.shape == (BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH)


def test_resblock(input_tensor: torch.Tensor) -> None:
    res_block = ResBlock(IN_CHANNELS, OUT_CHANNELS)
    output = res_block(input_tensor)
    assert output.shape == (BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH)


def test_downresblock(input_tensor: torch.Tensor) -> None:
    down_res_block = DownResBlock(IN_CHANNELS, OUT_CHANNELS)
    output = down_res_block(input_tensor)
    assert output.shape == (BATCH_SIZE, OUT_CHANNELS, HEIGHT // 2, WIDTH // 2)


def test_upresblock(input_tensor: torch.Tensor) -> None:
    up_res_block = UpResBlock(IN_CHANNELS, OUT_CHANNELS)
    output = up_res_block(input_tensor)
    assert output.shape == (BATCH_SIZE, OUT_CHANNELS, HEIGHT * 2, WIDTH * 2)


if __name__ == "__main__":
    pytest.main([__file__])
