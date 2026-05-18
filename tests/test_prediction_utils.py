import json

from nn.prediction_utils import prediction_key, write_target_predictions


def test_prediction_key_matches_existing_all_imgs_compatibility():
    assert prediction_key("all_imgs/sample.jpg") == "sample.jpg"
    assert prediction_key("nested/all_imgs/sample.jpg") == "nested/sample.jpg"


def test_write_target_predictions_preserves_submission_shape(tmp_path):
    output_path = tmp_path / "predictions.json"

    write_target_predictions({"sample.jpg": 4}, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"target": {"sample.jpg": 4}}
