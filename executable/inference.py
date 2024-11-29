from pathlib import Path

import argh
import cv2
import numpy as np
import tifffile
import torch
from config.config import Config
from model.model import CoralModel

Pathlike = Path | str
DEFAULT_SIZE = 2048


def main(model_path: Pathlike, image_path: Pathlike, size: int = DEFAULT_SIZE) -> None:
    """
    Run inference using a saved model checkpoint
    """
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    if image_path.suffix.lower() in (".tif", ".tiff"):
        image = tifffile.imread(image_path)
    else:
        print(image_path)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(image.shape)
    image = image[:size, :size, :3]
    print(image.shape)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = image.float() / 255

    config = Config()
    model = CoralModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()

    image = image.to(device=model.device)
    with torch.no_grad():
        pred = model(image)

    print(pred)


if __name__ == "__main__":
    argh.dispatch_command(main)
