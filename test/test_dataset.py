from pathlib import Path

import pytest
import torch
from config.config import Config
from dataset.dataset import CoralDataset, Sample


@pytest.fixture
def config() -> Config:
    current_dir = Path(__file__).parent
    return Config(data_dir=current_dir / "data" / "dataset", split="train")


@pytest.fixture
def coral_dataset(config: Config) -> CoralDataset:
    return CoralDataset(config)


def test_coral_dataset_getitem(coral_dataset: CoralDataset) -> None:
    assert len(coral_dataset) > 0, "Dataset should not be empty"
    sample_dict = coral_dataset[0]
    assert isinstance(sample_dict, dict), "Dataset item should be a dict"
    sample = Sample.from_dict(sample_dict)
    assert isinstance(sample.image, torch.Tensor), "Image should be a torch.Tensor"
    assert isinstance(sample.mask, torch.Tensor), "Mask should be a torch.Tensor"

    assert sample.image.dtype == torch.float32, "Image should be a float tensor"
    assert sample.image.ndim == 3, "Image should have 3 dimensions (C, H, W)"
    assert sample.image.shape[0] == 3, "Image should have 3 channels"
    assert (
        0 <= sample.image.min() <= sample.image.max() <= 1
    ), "Image values should be in range [0, 1]"

    assert sample.mask.dtype == torch.float32, "Mask should be a float tensor"
    assert sample.mask.ndim == 3, "Mask should have 3 dimensions (C, H, W)"
    assert sample.mask.shape[0] == 1, "Mask should have 1 channel"


if __name__ == "__main__":
    pytest.main([__file__])
