"""
Reshape raw predictions into the competition submission format.

Maps image-name-keyed predictions to idx_test-keyed predictions
as required by the nuwe.io submission format.

Usage:
    python -m postprocessing.reshape_solution \
        --predictions raw_predictions.json --test test.csv --output predictions.json
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def reshape(predictions_path: str, test_csv_path: str, output_path: str = "predictions.json"):
    with Path(predictions_path).open(encoding="utf-8") as f:
        data = json.load(f)

    df_test = pd.read_csv(test_csv_path)
    prediction_count = len(data["target"])
    test_count = len(df_test)

    if prediction_count != test_count:
        raise ValueError(f"Prediction count mismatch: {prediction_count} predictions vs {test_count} test samples")

    # Vectorized mapping: image path -> idx_test -> prediction
    df_test["img_key"] = df_test["path_img"].str.replace("all_imgs/", "", regex=False)
    df_test["prediction"] = df_test["img_key"].map(data["target"])

    missing = df_test["prediction"].isna().sum()
    if missing > 0:
        missing_keys = df_test.loc[df_test["prediction"].isna(), "img_key"].tolist()[:5]
        raise ValueError(
            f"{missing} images have no matching prediction. "
            f"First unmatched keys: {missing_keys}"
        )

    new_target = dict(zip(df_test["idx_test"].astype(str), df_test["prediction"].astype(int)))

    output = {"target": new_target}
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(output, f)

    logger.info(f"Reshaped {len(new_target)} predictions -> {output_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Reshape predictions to submission format")
    parser.add_argument("--predictions", type=str, required=True, help="Raw predictions JSON")
    parser.add_argument("--test", type=str, required=True, help="Path to test.csv")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output submission JSON")
    args = parser.parse_args()

    reshape(args.predictions, args.test, args.output)
