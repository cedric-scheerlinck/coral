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
    assert hasattr(model, "loss_fn")
    assert hasattr(model, "config")
    output = model(image)
    loss = model.loss_fn(output, target)
    assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__])
