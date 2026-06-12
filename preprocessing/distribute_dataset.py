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
        img_path = base / row["path_img"]
        label = str(row["label"])
        dest_dir = train_dir / label
        dest_dir.mkdir(exist_ok=True)
        shutil.copy(str(img_path), str(dest_dir))

    logger.info(f"Copied {len(df_train)} training images to {train_dir}")

    # Copy test images to flat directory
    for _, row in df_test.iterrows():
        img_path = base / row["path_img"]
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
