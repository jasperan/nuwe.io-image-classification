import json

import numpy as np


def test_tta_predict_writes_target_predictions_with_compatible_image_keys(
    tmp_path, monkeypatch, fake_inference_modules
):
    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

    module = fake_inference_modules("predict_new", FakeYOLO, with_pil=True)
    monkeypatch.setattr(module, "predict_with_tta", lambda model, image_path, imgsz=640: np.array([0.1, 0.7, 0.2]))
    (tmp_path / "test.csv").write_text("path_img\nall_imgs/sample.jpg\n", encoding="utf-8")
    output_path = tmp_path / "predictions_tta.json"

    module.predict("model.pt", str(tmp_path), str(output_path))

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"target": {"sample.jpg": 1}}


def test_tta_predict_supports_package_import(fake_inference_modules):
    class FakeYOLO:
        pass

    module = fake_inference_modules("predict_new", FakeYOLO, package=True, with_pil=True)

    assert callable(module.predict)
    assert module.prediction_key("all_imgs/sample.jpg") == "sample.jpg"
