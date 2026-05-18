"""
Predict class labels for test images using a trained YOLO classification model.

Reads test.csv, runs inference on each image, and saves predictions to JSON.
"""

import logging
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

if __package__:
    from .prediction_utils import prediction_key, write_target_predictions
else:
    from prediction_utils import prediction_key, write_target_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def predict(model_path: str, base_path: str, output_path: str = "predictions.json"):
    model = YOLO(model_path)
    base = Path(base_path)

    df_test = pd.read_csv(base / "test.csv", sep=",")
    logger.info(f"Test dataset length: {len(df_test)}")

    predictions = {}
    for idx, row in df_test.iterrows():
        img_rel = row["path_img"]
        img_path = base / img_rel
        results = model(str(img_path))
        top1_class = int(results[0].probs.top1)
        img_key = prediction_key(img_rel)
        predictions[img_key] = top1_class
        logger.info(f"[{idx + 1}/{len(df_test)}] {img_key} -> class {top1_class}")

    write_target_predictions(predictions, output_path)
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO classification inference")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--data", type=str, required=True, help="Base path containing test.csv and images")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON path")
    args = parser.parse_args()

    predict(args.model, args.data, args.output)
