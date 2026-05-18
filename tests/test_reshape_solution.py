import json

import pytest

from postprocessing.reshape_solution import reshape


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_reshape_maps_image_predictions_to_submission_ids(tmp_path):
    predictions_path = tmp_path / "raw_predictions.json"
    test_csv_path = tmp_path / "test.csv"
    output_path = tmp_path / "submission.json"
    write_json(predictions_path, {"target": {"first.jpg": 7, "second.jpg": 2}})
    test_csv_path.write_text(
        "idx_test,path_img\n101,all_imgs/first.jpg\n102,all_imgs/second.jpg\n",
        encoding="utf-8",
    )

    reshape(str(predictions_path), str(test_csv_path), str(output_path))

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "target": {"101": 7, "102": 2},
    }


def test_reshape_rejects_prediction_count_mismatch(tmp_path):
    predictions_path = tmp_path / "raw_predictions.json"
    test_csv_path = tmp_path / "test.csv"
    output_path = tmp_path / "submission.json"
    write_json(predictions_path, {"target": {"first.jpg": 7}})
    test_csv_path.write_text(
        "idx_test,path_img\n101,all_imgs/first.jpg\n102,all_imgs/second.jpg\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Prediction count mismatch"):
        reshape(str(predictions_path), str(test_csv_path), str(output_path))


def test_reshape_rejects_missing_prediction_keys(tmp_path):
    predictions_path = tmp_path / "raw_predictions.json"
    test_csv_path = tmp_path / "test.csv"
    output_path = tmp_path / "submission.json"
    write_json(predictions_path, {"target": {"first.jpg": 7, "missing-count-filler.jpg": 0}})
    test_csv_path.write_text(
        "idx_test,path_img\n101,all_imgs/first.jpg\n102,all_imgs/second.jpg\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="images have no matching prediction"):
        reshape(str(predictions_path), str(test_csv_path), str(output_path))
