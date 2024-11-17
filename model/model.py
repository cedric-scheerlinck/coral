import typing as T

import pytorch_lightning as pl
import torch
from config.config import Config
from dataset.dataset import Sample
from loss.loss import BCELoss
from loss.loss import DiceLoss
from network.network import UNet
from util.image_util import resize


class CoralModel(pl.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.network = UNet()
        self.losses = [BCELoss(), DiceLoss()]
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_pred(self, image: torch.Tensor) -> torch.Tensor:
        orig_device = image.device
        image = image.to(self.device)
        ndim = image.ndim
        if ndim == 3:
            image = image.unsqueeze(0)
        pred = self.network(image)
        pred = resize(pred, image)
        pred = pred.sigmoid()
        if ndim == 3:
            pred = pred.squeeze(0)
        return pred.to(orig_device).detach()

    #    def train_dataloader(self) -> T.Any:
    #        indices = torch.randperm(len(self.dataset)).tolist()[:self.config.subset_length]
    #        sampler = SubsetRandomSampler(indices)
    #        return DataLoader(self.dataset)
    #

    def training_step(self, data_blob: dict, batch_idx: int) -> torch.Tensor:
        sample = Sample.from_dict(data_blob)
        preds = self.network(sample.image)
        loss = 0
        for loss_fn in self.losses:
            loss_val = loss_fn(preds, sample.mask)
            loss += loss_val
            self.log(
                f"train/{loss_fn.__class__.__name__}",
                loss_val,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx % self.config.log_images_every_n_steps == 0:
            self.log_images(sample, preds.sigmoid(), "train")
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
