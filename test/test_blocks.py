import pytest
import torch
from network.blocks import DownResBlock
from network.blocks import ResBlock
from network.blocks import UpResBlock
from network.blocks import conv_bn


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def in_channels() -> int:
    return 8


@pytest.fixture
def out_channels() -> int:
    return 16


@pytest.fixture
def height() -> int:
    return 32


@pytest.fixture
def width() -> int:
    return 32


@pytest.fixture
def input_tensor(
    batch_size: int, in_channels: int, height: int, width: int
) -> torch.Tensor:
    return torch.randn(batch_size, in_channels, height, width)


def test_conv_bn(
    input_tensor: torch.Tensor,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
) -> None:
    conv_bn_layer = conv_bn(in_channels, out_channels)
    out = conv_bn_layer(input_tensor)
    assert out.shape == (batch_size, out_channels, height, width)


def test_resblock(
    input_tensor: torch.Tensor,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
) -> None:
    res_block = ResBlock(in_channels, out_channels)
    output = res_block(input_tensor)
    assert output.shape == (batch_size, out_channels, height, width)


def test_downresblock(
    input_tensor: torch.Tensor,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
) -> None:
    down_res_block = DownResBlock(in_channels, out_channels)
    output = down_res_block(input_tensor)
    assert output.shape == (batch_size, out_channels, height // 2, width // 2)


def test_upresblock(
    input_tensor: torch.Tensor,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
) -> None:
    up_res_block = UpResBlock(in_channels, out_channels)
    output = up_res_block(input_tensor)
    assert output.shape == (batch_size, out_channels, height * 2, width * 2)


if __name__ == "__main__":
    pytest.main([__file__])
