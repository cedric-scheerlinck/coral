import cv2
import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    cv2.setRNGSeed(seed)
