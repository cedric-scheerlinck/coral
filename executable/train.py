# flake8: noqa: E402
from util.disable_multithreading import disable_multithreading

disable_multithreading()

from pathlib import Path

import pytorch_lightning as pl
from config.config import Config
from dataset.dataset import CoralDataset
from model.model import CoralModel
from torch.utils.data import DataLoader
from util.get_config import get_config
from util.rand_seed import set_random_seed

set_random_seed()


def main(config: Config) -> None:
    # Create output directory if it doesn't exist
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create datasets and dataloaders
    train_dataset = CoralDataset(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    # Initialize model
    model = CoralModel(config)

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        default_root_dir=config.output_dir,
        log_every_n_steps=config.log_every_n_steps,
    )

    # Train the model
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    config = get_config()
    main(config)
