import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def import_nn_predict(monkeypatch, fake_yolo):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "nn"))
    sys.modules.pop("nn_predict", None)
    return importlib.import_module("nn_predict")


def import_nn_predict_package(monkeypatch, fake_yolo):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    sys.modules.pop("nn.nn_predict", None)
    return importlib.import_module("nn.nn_predict")


def test_predict_writes_target_predictions_with_compatible_image_keys(tmp_path, monkeypatch):
    calls = []

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def __call__(self, image_path):
            calls.append(image_path)
            return [types.SimpleNamespace(probs=types.SimpleNamespace(top1=3))]

    module = import_nn_predict(monkeypatch, FakeYOLO)
    (tmp_path / "test.csv").write_text("path_img\nall_imgs/sample.jpg\n", encoding="utf-8")
    output_path = tmp_path / "predictions.json"

    module.predict("model.pt", str(tmp_path), str(output_path))

    assert calls == [str(tmp_path / "all_imgs/sample.jpg")]
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"target": {"sample.jpg": 3}}


def test_predict_supports_package_import(monkeypatch):
    class FakeYOLO:
        pass

    module = import_nn_predict_package(monkeypatch, FakeYOLO)

    assert callable(module.predict)
    assert module.prediction_key("all_imgs/sample.jpg") == "sample.jpg"
