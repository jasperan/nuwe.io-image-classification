import json
import types


def test_predict_writes_target_predictions_with_compatible_image_keys(
    tmp_path, fake_inference_modules
):
    calls = []

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def __call__(self, image_path):
            calls.append(image_path)
            return [types.SimpleNamespace(probs=types.SimpleNamespace(top1=3))]

    module = fake_inference_modules("nn_predict", FakeYOLO)
    (tmp_path / "test.csv").write_text("path_img\nall_imgs/sample.jpg\n", encoding="utf-8")
    output_path = tmp_path / "predictions.json"

    module.predict("model.pt", str(tmp_path), str(output_path))

    assert calls == [str(tmp_path / "all_imgs/sample.jpg")]
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"target": {"sample.jpg": 3}}
