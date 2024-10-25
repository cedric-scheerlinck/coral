# import pytorch dataset
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import torch
from torch.utils.data import Dataset

BASE_DIR = Path("/media/cedric/Storage1/coral_data/dataset")


@dataclass
class Sample:
    image: torch.Tensor  # shape (3, H, W), uint8
    mask: torch.Tensor  # shape (1, H, W), bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(**data)


class CoralDataset(Dataset):
    def __init__(self, dataset_dir: Path = BASE_DIR) -> None:
        self.dataset_dir = dataset_dir
        self.sample_paths = sorted(dataset_dir.glob("**/*.png"))

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.sample_paths[idx]
        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGBA)
        image = torch.from_numpy(image_np).permute(2, 0, 1)
        image, mask = image.split([3, 1], dim=0)
        mask = ~mask.bool()
        image = image.float() / 255.0
        return Sample(image=image, mask=mask).to_dict()
