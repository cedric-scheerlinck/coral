import pytorch_lightning as pl
import torch
from dataset.dataset import Sample
from loss.loss import SegmentationLoss
from network.network import UNet


class CoralSegmentationModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = UNet()
        self.loss_fn = SegmentationLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
