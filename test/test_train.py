from pathlib import Path

import pytest
from config.config import Config
from executable.train import main


@pytest.fixture
def config() -> Config:
    current_dir = Path(__file__).parent
    return Config(
        data_dir=current_dir / "data" / "dataset",
        output_dir=current_dir / "data" / "logs",
        split="train",
        batch_size=1,
        num_epochs=1,
        num_workers=1,
        log_image_every_n_steps=1,
    )


def test_main(config):
    main(config)


if __name__ == "__main__":
    pytest.main([__file__])
