from pathlib import Path


def get_sample_paths(dataset_dir: Path) -> list[Path]:
    meta_file = dataset_dir / "meta.txt"
    if meta_file.is_file():
        with open(meta_file, "r") as f:
            return [dataset_dir / line.strip() for line in f.readlines()]
    else:
        sample_paths = sorted(dataset_dir.glob("**/*.png"))
        # write to meta.txt
        with open(meta_file, "w") as f:
            for path in sample_paths:
                f.write(f"{path.relative_to(dataset_dir)}\n")
        return sample_paths
