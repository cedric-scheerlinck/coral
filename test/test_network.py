import pytest
import torch
from network.network import UNet


@pytest.fixture
def unet() -> UNet:
    return UNet()


def test_unet_forward(unet: UNet) -> None:
    # Test input
    batch_size = 2
    channels = 3
    height = 32
    width = 32
    x = torch.randn(batch_size, channels, height, width)

    # Forward pass
    output = unet(x)

    # Check output shape
    expected_output_height = height // 4  # Due to 2 down-sampling operations
    expected_output_width = width // 4
    expected_output_channels = 1  # As defined in the coral_head

    assert output.shape == (
        batch_size,
        expected_output_channels,
        expected_output_height,
        expected_output_width,
    )


if __name__ == "__main__":
    pytest.main([__file__])
