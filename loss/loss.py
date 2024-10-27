import torch
import torch.nn as nn
import torch.nn.functional as F
from util.image_util import max_pool


class BCELoss(nn.Module):
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = max_pool(gt, pred)
        pos_weight = torch.tensor(100.0, device=pred.device)
        return F.binary_cross_entropy_with_logits(pred, gt, pos_weight=pos_weight)


class DiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = max_pool(gt, pred)
        return dice_loss(pred, gt)


def dice_loss(
    pred: torch.Tensor, gt: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    pred_f = pred.flatten().sigmoid()
    gt_f = gt.flatten()
    intersection = (pred_f * gt_f).sum()
    union = pred_f.sum() + gt_f.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)
