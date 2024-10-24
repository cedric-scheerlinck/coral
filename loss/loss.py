import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(pred, gt.float())
