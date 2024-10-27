import pytest
import torch
from config.config import Config
from model.model import CoralModel

BATCH_SIZE = 1
HEIGHT = 64
WIDTH = 64


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def image() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, 3, HEIGHT, WIDTH)


@pytest.fixture
def target() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, 1, HEIGHT, WIDTH)


def test_coral_model(config: Config, image: torch.Tensor, target: torch.Tensor) -> None:
    model = CoralModel(config)
    assert hasattr(model, "network")
    assert hasattr(model, "losses")
    assert hasattr(model, "config")
    output = model(image)
    assert output.ndim == 4
    assert output.shape[0] == BATCH_SIZE
    assert output.shape[1] == 1


if __name__ == "__main__":
    pytest.main([__file__])
