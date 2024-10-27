import typing as T
from dataclasses import dataclass
from pathlib import Path

from util.constants import Split

Pathlike = T.Union[str, Path]
DATA_DIR = "/media/cedric/Storage1/coral_data/dataset"


@dataclass
class Config:
    data_dir: Pathlike = DATA_DIR
    split: Split = "train"
    max_num_samples: int = -1

    output_dir: Pathlike = ""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 4
    num_epochs: int = 100
    num_workers: int = 4
    log_image_every_n_steps: int = 1000
