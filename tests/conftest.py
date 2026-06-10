import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

NN_DIR = Path(__file__).resolve().parents[1] / "nn"


def _install_fake_pil(monkeypatch):
    fake_pil = types.ModuleType("PIL")
    fake_image = types.ModuleType("PIL.Image")
    fake_pil.Image = fake_image
    fake_image.FLIP_LEFT_RIGHT = object()
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image)


def _install_fake_ultralytics(monkeypatch, fake_yolo):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))


@pytest.fixture
def fake_inference_modules(monkeypatch):
    """Stub out ultralytics/PIL and import a predict module either as a loose
    script (off the ``nn/`` dir) or as part of the ``nn`` package.

    Returns an ``import_module(name, fake_yolo, *, package=False, with_pil=False)``
    callable that yields the imported module.
    """

    def import_module(name, fake_yolo, *, package=False, with_pil=False):
        _install_fake_ultralytics(monkeypatch, fake_yolo)
        if with_pil:
            _install_fake_pil(monkeypatch)
        if package:
            module_name = f"nn.{name}"
        else:
            monkeypatch.syspath_prepend(str(NN_DIR))
            module_name = name
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)

    return import_module
