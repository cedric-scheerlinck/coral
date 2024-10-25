import pytorch_lightning as pl
import torch
from config.config import Config
from dataset.dataset import Sample
from loss.loss import SegmentationLoss
from network.network import UNet


class CoralModel(pl.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.network = UNet()
        self.loss_fn = SegmentationLoss()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, data_blob: dict, batch_idx: int) -> torch.Tensor:
        sample = Sample.from_dict(data_blob)
        predictions = self(sample.image)
        loss = self.loss_fn(predictions, sample.mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, data_blob: dict, batch_idx: int) -> torch.Tensor:
        sample = Sample.from_dict(data_blob)
        predictions = self(sample.image)
        loss = self.loss_fn(predictions, sample.mask)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )
        return optimizer
