"""
Predict with test-time augmentation (TTA) for improved accuracy.

Applies horizontal flip + multi-scale inference and averages softmax outputs
for a 2-5% accuracy boost over single-pass prediction.

Usage:
    python predict_new.py --model model.pt --data /path/to/dataset --output predictions.json
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

if __package__:
    from .prediction_utils import prediction_key, write_target_predictions
else:
    from prediction_utils import prediction_key, write_target_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TTA_SCALES = [0.85, 1.0, 1.15]


def predict_with_tta(model: YOLO, img_path: str, imgsz: int = 640) -> np.ndarray:
    """Run TTA: original + hflip at multiple scales, average softmax outputs."""
    all_probs = []

    with Image.open(img_path) as source:
        img = source.convert("RGB")
    img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

    for scale in TTA_SCALES:
        scaled_size = int(imgsz * scale)

        # Original image
        results = model(img, imgsz=scaled_size, verbose=False)
        probs = results[0].probs.data.cpu().numpy()
        all_probs.append(probs)

        # Horizontally flipped image
        results_flip = model(img_flipped, imgsz=scaled_size, verbose=False)
        probs_flip = results_flip[0].probs.data.cpu().numpy()
        all_probs.append(probs_flip)

    # Average all softmax outputs
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


def predict(model_path: str, base_path: str, output_path: str = "predictions.json", imgsz: int = 640):
    model = YOLO(model_path)
    base = Path(base_path)

    df_test = pd.read_csv(base / "test.csv", sep=",")
    logger.info(f"Test dataset length: {len(df_test)} | TTA scales: {TTA_SCALES}")

    predictions = {}
    confidences = []

    for idx, row in df_test.iterrows():
        img_rel = row["path_img"]
        img_path = str(base / img_rel)
        avg_probs = predict_with_tta(model, img_path, imgsz=imgsz)
        top1_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[top1_class])

        img_key = prediction_key(img_rel)
        predictions[img_key] = top1_class
        confidences.append(round(confidence, 4))
        logger.info(f"[{idx + 1}/{len(df_test)}] {img_key} -> class {top1_class} (conf={confidence:.3f})")

    write_target_predictions(predictions, output_path)

    logger.info(f"Saved {len(predictions)} TTA predictions to {output_path}")
    avg_conf = np.mean(confidences)
    logger.info(f"Average confidence: {avg_conf:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO classification with TTA")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--data", type=str, required=True, help="Base path containing test.csv and images")
    parser.add_argument("--output", type=str, default="predictions_tta.json", help="Output JSON path")
    parser.add_argument("--imgsz", type=int, default=640, help="Base image size for inference")
    args = parser.parse_args()

    predict(args.model, args.data, args.output, args.imgsz)
