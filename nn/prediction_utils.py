"""Shared helpers for competition prediction output."""

import json
from pathlib import Path
from typing import Mapping, Union

ALL_IMAGES_PREFIX = "all_imgs/"


def prediction_key(path_img: str) -> str:
    """Return the image key expected by the postprocessing step."""
    return path_img.replace(ALL_IMAGES_PREFIX, "")


def write_target_predictions(predictions: Mapping[str, int], output_path: Union[str, Path]) -> None:
    output = {"target": dict(predictions)}
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(output, f)
