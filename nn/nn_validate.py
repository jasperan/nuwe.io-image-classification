"""
Validate a trained YOLO classification model and print metrics.
"""

import logging

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def validate(model_path: str):
    model = YOLO(model_path)
    logger.info(f"Validating model: {model_path}")
    results = model.val()
    logger.info(f"Top-1 accuracy: {results.top1:.4f}")
    logger.info(f"Top-5 accuracy: {results.top5:.4f}")
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Validate YOLO classification model")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    args = parser.parse_args()

    validate(args.model)
