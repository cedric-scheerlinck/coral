import pytest
import torch
from loss.loss import BCELoss, DiceLoss

BATCH_SIZE = 2
CHANNELS = 1
HEIGHT = 32
WIDTH = 32


@pytest.fixture
def pred() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture
def target() -> torch.Tensor:
    return torch.randint(0, 2, (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)).float()


@pytest.mark.parametrize("loss_class", [BCELoss, DiceLoss])
def test_loss(loss_class, pred: torch.Tensor, target: torch.Tensor):
    loss_fn = loss_class()

    # Ensure pred requires gradients
    pred.requires_grad_(True)

    loss = loss_fn(pred, target)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0

    # Check if loss requires gradients
    assert loss.requires_grad, "Loss tensor doesn't require gradients"

    # Print loss grad_fn for debugging
    print(f"Loss grad_fn: {loss.grad_fn}")

    loss.backward()
    assert pred.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])
