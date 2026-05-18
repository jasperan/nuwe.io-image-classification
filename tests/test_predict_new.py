import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def import_predict_new(monkeypatch, fake_yolo):
    fake_pil = types.ModuleType("PIL")
    fake_image = types.ModuleType("PIL.Image")
    fake_pil.Image = fake_image
    fake_image.FLIP_LEFT_RIGHT = object()
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "nn"))
    sys.modules.pop("predict_new", None)
    return importlib.import_module("predict_new")


def import_predict_new_package(monkeypatch, fake_yolo):
    fake_pil = types.ModuleType("PIL")
    fake_image = types.ModuleType("PIL.Image")
    fake_pil.Image = fake_image
    fake_image.FLIP_LEFT_RIGHT = object()
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    sys.modules.pop("nn.predict_new", None)
    return importlib.import_module("nn.predict_new")


def test_tta_predict_writes_target_predictions_with_compatible_image_keys(tmp_path, monkeypatch):
    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

    module = import_predict_new(monkeypatch, FakeYOLO)
    monkeypatch.setattr(module, "predict_with_tta", lambda model, image_path, imgsz=640: np.array([0.1, 0.7, 0.2]))
    (tmp_path / "test.csv").write_text("path_img\nall_imgs/sample.jpg\n", encoding="utf-8")
    output_path = tmp_path / "predictions_tta.json"

    module.predict("model.pt", str(tmp_path), str(output_path))

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"target": {"sample.jpg": 1}}


def test_tta_predict_supports_package_import(monkeypatch):
    class FakeYOLO:
        pass

    module = import_predict_new_package(monkeypatch, FakeYOLO)

    assert callable(module.predict)
    assert module.prediction_key("all_imgs/sample.jpg") == "sample.jpg"
