import typing as T
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import torch
from config.config import Config
from torch.utils.data import Dataset

from dataset.util import get_sample_paths

BLOCKLIST = {
    "T12_tp2_13.12.22",
    "T3_22.12.19",
    "T18_tp8_31.01.23",
    "T18_tp7_24.01.23",
    "T14_tp1_06.12.22",
    "T7_tp3_19.12.22",
    "T10_tp2_13.12.22",
}


@dataclass
class Sample:
    image: torch.Tensor  # shape (3, H, W)
    mask: torch.Tensor  # shape (1, H, W)
    path: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(**data)


class CoralDataset(Dataset):
    def __init__(self, config: Config) -> None:
        self.data_dir = Path(config.data_dir) / config.split
        assert self.data_dir.is_dir(), "Dataset directory does not exist"
        sample_paths = get_sample_paths(self.data_dir)
        self.sample_paths = self.filter_paths(sample_paths)
        if config.max_num_samples >= 0:
            self.sample_paths = self.sample_paths[: config.max_num_samples]

    def filter_paths(self, sample_paths: T.List[Path]) -> T.List[Path]:
        return [path for path in sample_paths if path.parent.name not in BLOCKLIST]

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.sample_paths[idx]
        return self.load_path(image_path)

    def load_path(self, image_path: Path) -> Sample:
        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGBA)
        image = torch.from_numpy(image_np).permute(2, 0, 1)
        image, mask = image.split([3, 1], dim=0)
        mask = (mask < 254).float()
        image = image.float() / 255.0
        return Sample(image=image, mask=mask, path=str(image_path)).to_dict()
