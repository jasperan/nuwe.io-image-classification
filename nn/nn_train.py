"""
Train a YOLO classification model with optimized augmentation settings.

Usage:
    python -m nn.nn_train --data /path/to/dataset --epochs 300 --imgsz 640
"""

import logging

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def train(data_path: str, epochs: int = 300, imgsz: int = 640, model_name: str = "yolov8x-cls.yaml"):
    model = YOLO(model_name)

    logger.info(f"Training {model_name} on {data_path} for {epochs} epochs at {imgsz}px")
    model.train(
        data=data_path,
        epochs=epochs,
        patience=75,
        imgsz=imgsz,
        verbose=True,
        augment=True,
        # Color augmentation (was all 0.0 — now enabled)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Geometric augmentation
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        # Advanced augmentation
        mosaic=0.5,
        mixup=0.15,
    )
    logger.info("Training complete")
    return model


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train YOLO classification model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--model", type=str, default="yolov8x-cls.yaml", help="Model architecture YAML")
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.model)
