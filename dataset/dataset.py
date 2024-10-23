# import pytorch dataset
from torch.utils.data import Dataset
from pathlib import Path
import typing as T

BASE_DIR = Path('/media/cedric/Storage1/coral_data/dataset')

class CoralDataset(Dataset):
    def __init__(self, dataset_dir: Path = BASE_DIR):
        self.dataset_dir = dataset_dir
        self.sample_paths = sorted(dataset_dir.glob('*.png'))

    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        image_path = self.sample_paths[idx]
        image = Image.open(image_path)
        return image
