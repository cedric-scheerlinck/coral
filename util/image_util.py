import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def get_ratio(x: torch.Tensor, ref: torch.Tensor) -> tuple[int, int]:
    return (x.shape[-2] // ref.shape[-2], x.shape[-1] // ref.shape[-1])


def avg_pool(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        ratio = get_ratio(x, ref)
        return F.avg_pool2d(x, kernel_size=ratio, stride=ratio)
    return x


def max_pool(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        ratio = get_ratio(x, ref)
        return F.max_pool2d(x, kernel_size=ratio, stride=ratio)
    return x


def min_pool(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return -max_pool(-x, ref)


def resize(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        return TF.resize(
            x, size=ref.shape[-2:], interpolation=TF.InterpolationMode.NEAREST
        )
    return x


def numpy_from_torch(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
