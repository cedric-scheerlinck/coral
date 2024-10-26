# flake8: noqa: E402
from util.disable_multithreading import disable_multithreading

disable_multithreading()

import argparse
from dataclasses import fields
from pathlib import Path

import pytorch_lightning as pl
from config.config import Config
from dataset.dataset import CoralDataset
from model.model import CoralModel
from torch.utils.data import DataLoader
from util.rand_seed import set_random_seed

OUTPUT_DIR = Path("/media/cedric/Storage1/coral_data/logs")

set_random_seed()


def get_config() -> Config:
    parser = argparse.ArgumentParser(description="Train the Coral Network")
    config = Config()
    for field in fields(config):
        parser.add_argument(
            f"--{field.name}",
            type=field.type,
            default=None,
            help=field.metadata.get("help", ""),
        )
    args = parser.parse_args()
    for field in fields(config):
        arg = getattr(args, field.name)
        if arg is not None:
            print(f"Overriding {field.name} -> {arg}")
            setattr(config, field.name, arg)

    if not config.output_dir:
        config.output_dir = OUTPUT_DIR

    return config


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
