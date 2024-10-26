import pytorch_lightning as pl
import torch
from config.config import Config
from dataset.dataset import Sample
from loss.loss import CoralLoss
from network.network import UNet


class CoralModel(pl.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.network = UNet()
        self.loss_fn = CoralLoss()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, data_blob: dict, batch_idx: int) -> torch.Tensor:
        sample = Sample.from_dict(data_blob)
        predictions = self.network(sample.image)
        loss = self.loss_fn(predictions, sample.mask)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx % self.config.log_every_n_steps == 0:
            self.log_images(sample, predictions, "train")
        return loss

    def log_images(self, sample: Sample, pred: torch.Tensor, split: str) -> None:
        for i in range(sample.image.shape[0]):
            self.logger.experiment.add_image(
                f"{split}_images/image_{i}", sample.image[i], self.global_step
            )
            self.logger.experiment.add_image(
                f"{split}_images/mask_{i}", sample.mask[i], self.global_step
            )
            self.logger.experiment.add_image(
                f"{split}_images/pred_{i}", pred[i], self.global_step
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return optimizer
