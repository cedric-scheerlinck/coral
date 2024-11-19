import re
from pathlib import Path

import argh
import cv2
import numpy as np
import tifffile
from roifile import ImagejRoi
from skimage.draw import polygon
from tqdm import tqdm

ROTATE = {
    "T1_22.12.06.tif": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "T1_22.12.13a.tif": cv2.ROTATE_90_CLOCKWISE,
    "T1_22.12.19a.tif": cv2.ROTATE_90_CLOCKWISE,
}

SKIP = {"T10_tp1_06.12.23.tif", "T17_tp4_02.01.23.tif", "T18_tp7_24.01.23.tif"}

SPLIT = {
    "test": {"T1_Done"},
    "val": {"T2_Done"},
}

TILE_SIZE = 1024  # Size of each tile (NxN)

DEFAULT_INPUT_PATH = Path(
    "/media/cedric/Storage1/coral_data/Phase_I/Phase_I_Stitched_Images"
)


def get_split_from_image_path(image_path: Path, base_path: Path) -> str:
    name = image_path.relative_to(base_path).parts[0]
    for split, names in SPLIT.items():
        if name in names:
            return split
    return "train"


def main(output: str, input_path: Path = DEFAULT_INPUT_PATH) -> None:
    all_image_paths = sorted(input_path.glob("**/*.tif"))

    for image_path in tqdm(all_image_paths):
        if image_path.name in SKIP:
            print(f"Skipping {image_path}")
            continue
        split = get_split_from_image_path(image_path, input_path)
        relative_path = image_path.relative_to(input_path)
        output_dir = Path(output) / split / relative_path
        if output_dir.suffix == ".tif":
            output_dir = output_dir.with_suffix("")
        if output_dir.exists():
            continue
        roi_dir = get_roi_dir_from_image_path(image_path)
        if roi_dir is None:
            print(f"No ROI directory found for {image_path}")
            continue
        # Load the image using tifffile
        full_image = tifffile.imread(image_path)

        if image_path.name in ROTATE:
            full_image = cv2.rotate(full_image, ROTATE[image_path.name])

        h, w, _ = full_image.shape

        # Create the mask
        full_mask = create_mask_from_roi(roi_dir, full_image.shape)

        # replace the 4th channel with the mask
        alpha = 255 - full_mask * 127
        if full_image.shape[2] == 4:
            full_image[:, :, 3] = alpha
        elif full_image.shape[2] == 3:
            full_image = np.dstack((full_image, alpha))
        else:
            print(f"Unsupported number of channels: {full_image.shape[2]}")
            continue

        # Create and save tiles
        create_and_save_tiles(full_image, TILE_SIZE, output_dir)


def create_mask_from_roi(
    roi_dir: Path, image_shape: tuple[int, int, int]
) -> np.ndarray:
    # Initialize mask as a 2D array
    roi_files = sorted(roi_dir.glob("*.roi"))
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    height, width = image_shape[:2]

    for roi_file in roi_files:
        roi = ImagejRoi.fromfile(roi_file)

        # Get the coordinates of the ROI
        coords = roi.coordinates()
        if coords is None or len(coords) == 0:
            continue  # Skip if no coordinates

        # Convert coordinates to NumPy array
        coords = np.array(coords, dtype=np.intp)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        # Ensure coordinates are within image bounds
        if x_coords.max() >= width or y_coords.max() >= height:
            print(f"Coordinates exceed image dimensions in {roi_file}")
            continue  # Skip or adjust the coordinates

        # Fill the polygon in the mask
        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
        mask[rr, cc] = 1  # Assign a label value (e.g., 1)

    return mask


def create_and_save_tiles(image: np.ndarray, tile_size: int, output_dir: Path) -> None:
    h, w, _ = image.shape
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            # Extract tile from image and mask
            tile_img = image[i : i + tile_size, j : j + tile_size]

            # Skip tiles that are smaller than the desired size
            if tile_img.shape[:2] != (tile_size, tile_size):
                continue

            # use cv2 to save lossless as png
            # convert to uint8 bgr
            tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGBA2BGRA).astype(np.uint8)
            cv2.imwrite(output_dir / f"{count:05d}.png", tile_img)
            count += 1


def get_roi_base_dir(image_path: Path) -> Path | None:
    for i in range(3):
        base_dir = image_path.parents[i]
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        roi_candidates = [d for d in subdirs if "roi" in d.name.lower()]
        if len(roi_candidates) == 1:
            return roi_candidates[0]
    return None


def get_roi_dir_from_image_path(image_path: Path) -> Path | None:
    name = image_path.name.removesuffix(".tif")
    roi_base_dir = get_roi_base_dir(image_path)
    if roi_base_dir is None:
        return None

    roi_dir = roi_base_dir / name
    if roi_dir.is_dir():
        return roi_dir

    tp_group = extract_tp_group(name)
    if tp_group is None:
        return None
    roi_candidate_dirs = [d for d in roi_base_dir.glob(f"*{tp_group}*") if d.is_dir()]
    if roi_candidate_dirs:
        return roi_candidate_dirs[0]
    return None


def extract_tp_group(text: str) -> str | None:
    if match := re.search(r"tp[0-9]+", text.lower()):
        return match.group()
    return None


if __name__ == "__main__":
    argh.dispatch_command(main)
