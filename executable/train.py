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


class CheckpointEveryEpoch(pl.Callback):
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        trainer.save_checkpoint(f"{trainer.log_dir}/checkpoints/{epoch:04d}.ckpt")


def main(config: Config) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = CoralDataset(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    model = CoralModel(config)

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        default_root_dir=config.output_dir,
        log_every_n_steps=config.log_every_n_steps,
    )
    print(f"logging to {config.output_dir}")
    trainer.callbacks.append(CheckpointEveryEpoch())
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    config = get_config()
    main(config)
