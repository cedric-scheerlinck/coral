import torch
import torch.nn.functional as F


def get_ratio(gt: torch.Tensor, pred: torch.Tensor) -> tuple[int, int]:
    return (gt.shape[-2] // pred.shape[-2], gt.shape[-1] // pred.shape[-1])


def avg_pool(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if gt.shape[-2:] != pred.shape[-2:]:
        ratio = get_ratio(gt, pred)
        return F.avg_pool2d(gt, kernel_size=ratio, stride=ratio)
    return gt


def max_pool(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if gt.shape[-2:] != pred.shape[-2:]:
        ratio = get_ratio(gt, pred)
        return F.max_pool2d(gt, kernel_size=ratio, stride=ratio)
    return gt


def min_pool(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return -max_pool(-gt, pred)
