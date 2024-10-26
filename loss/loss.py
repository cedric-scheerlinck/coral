import torch
import torch.nn as nn
import torch.nn.functional as F
from util.image_util import max_pool


class CoralLoss(nn.Module):
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = max_pool(gt, pred)
        return F.binary_cross_entropy_with_logits(pred, gt)
