"""
Distribute dataset images into class-labeled directories for YOLO training.

Reads train.csv and test.csv, copies images into train/{label}/ and test/ directories
following the Ultralytics classification dataset structure.

Usage:
    python -m preprocessing.distribute_dataset --data /path/to/dataset --output /path/to/output
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_LABELS = frozenset(str(index) for index in range(8))


def _manifest_field(row, name: str) -> str:
    """Return a manifest field as text, rejecting empty or missing values."""
    value = row[name]
    if pd.isna(value):
        raise ValueError(f"{name} must be present in every manifest row")
    return str(value)


def _resolve_within(root: Path, value: str, field: str) -> Path:
    """Resolve a manifest path against ``root``, rejecting absolute paths and escapes."""
    candidate = Path(value)
    if not candidate.parts or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must be a relative path without '..': {value!r}")
    root_resolved = root.resolve()
    resolved = (root_resolved / candidate).resolve()
    if root_resolved not in resolved.parents:
        raise ValueError(f"{field} escapes {root_resolved}: {value!r}")
    return resolved


def _resolve_label(train_dir: Path, label: str) -> Path:
    """Resolve a class directory under ``train_dir`` for one of the known labels."""
    if label not in ALLOWED_LABELS:
        raise ValueError(f"label must be one of {sorted(ALLOWED_LABELS)}: {label!r}")
    train_resolved = train_dir.resolve()
    dest_dir = (train_resolved / label).resolve()
    if train_resolved not in dest_dir.parents:
        raise ValueError(f"label escapes {train_resolved}: {label!r}")
    return dest_dir


def distribute(data_path: str, output_path: str):
    base = Path(data_path)
    output = Path(output_path)

    df_train = pd.read_csv(base / "train.csv", sep=",")
    df_test = pd.read_csv(base / "test.csv", sep=",")
    logger.info(f"Dataset lengths: {len(df_train)} train | {len(df_test)} test")

    # Create output directories
    train_dir = output / "train"
    test_dir = output / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Distribute training images into class folders
    for _, row in df_train.iterrows():
        img_path = _resolve_within(base, _manifest_field(row, "path_img"), "path_img")
        label = _manifest_field(row, "label")
        dest_dir = _resolve_label(train_dir, label)
        dest_dir.mkdir(exist_ok=True)
        shutil.copy(str(img_path), str(dest_dir))

    logger.info(f"Copied {len(df_train)} training images to {train_dir}")

    # Copy test images to flat directory
    for _, row in df_test.iterrows():
        img_path = _resolve_within(base, _manifest_field(row, "path_img"), "path_img")
        shutil.copy(str(img_path), str(test_dir))

    logger.info(f"Copied {len(df_test)} test images to {test_dir}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Distribute dataset into YOLO format")
    parser.add_argument("--data", type=str, required=True, help="Base path with train.csv, test.csv, and images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for organized dataset")
    args = parser.parse_args()

    distribute(args.data, args.output)
