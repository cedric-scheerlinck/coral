import typing as T
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
        image_np = tifffile.imread(image_path)
    else:
        image_np = cv2.imread(str(image_path))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Split image into chunks of size x size
    h, w = image_np.shape[:2]
    n_rows = h // size
    n_cols = w // size

    config = Config()
    model = CoralModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model = model.to(device="cpu")

    image_np = image_np[..., :3]
    for i in range(n_rows):
        for j in range(n_cols):
            chunk = image_np[i * size : (i + 1) * size, j * size : (j + 1) * size]
            pred = get_pred(chunk, model)
            save_pred(image_path, pred, chunk, i * n_cols + j)


def save_pred(
    image_path: Pathlike,
    pred: np.ndarray,
    image: np.ndarray,
    idx: int,
    color: T.Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    output_path = image_path.parent / "pred" / f"{image_path.stem}_{idx:05d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contours, _ = cv2.findContours(
        (pred.squeeze() > 0.5).astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image = image.copy()
    cv2.drawContours(image, contours, -1, color, thickness=thickness)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image)
    print(f"Saved prediction to {output_path}")


def get_pred(chunk: np.ndarray, model: CoralModel) -> np.ndarray:
    image = torch.from_numpy(chunk)
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = image.float() / 255

    with torch.no_grad():
        pred = model.get_pred(image)

    pred = pred.squeeze().clamp(0, 1).cpu().numpy()
    return pred


if __name__ == "__main__":
    argh.dispatch_command(main)
