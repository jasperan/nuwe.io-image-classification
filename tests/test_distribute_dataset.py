import shutil
from pathlib import Path

import pytest

from preprocessing.distribute_dataset import distribute


def _write_bundle(data_dir: Path, train_rows: str, test_rows: str) -> None:
    (data_dir / "imgs").mkdir(parents=True)
    (data_dir / "imgs" / "first.png").write_bytes(b"first-image")
    (data_dir / "imgs" / "second.png").write_bytes(b"second-image")
    (data_dir / "train.csv").write_text("path_img,label\n" + train_rows, encoding="utf-8")
    (data_dir / "test.csv").write_text("path_img\n" + test_rows, encoding="utf-8")


def _outside_files(output_dir: Path, data_dir: Path) -> list:
    return [
        path
        for path in Path(output_dir).parent.rglob("*")
        if path.is_file() and output_dir not in path.parents and data_dir not in path.parents
    ]


def test_distribute_copies_bundle_into_labeled_directories(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    _write_bundle(data_dir, "imgs/first.png,0\nimgs/second.png,7\n", "imgs/second.png\n")

    distribute(str(data_dir), str(output_dir))

    assert (output_dir / "train" / "0" / "first.png").read_bytes() == b"first-image"
    assert (output_dir / "train" / "7" / "second.png").read_bytes() == b"second-image"
    assert (output_dir / "test" / "second.png").read_bytes() == b"second-image"
    assert _outside_files(output_dir, data_dir) == []


@pytest.mark.parametrize("label", ["../../escaped_relative", "/tmp/absolute_label", "../one_up"])
def test_distribute_rejects_labels_that_escape_the_output_directory(tmp_path, label):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    _write_bundle(data_dir, f"imgs/first.png,0\nimgs/second.png,{label}\n", "imgs/second.png\n")

    with pytest.raises(ValueError, match="label"):
        distribute(str(data_dir), str(output_dir))

    assert not (tmp_path / "escaped_relative").exists()
    assert not (tmp_path / "one_up").exists()
    assert _outside_files(output_dir, data_dir) == []


def test_distribute_rejects_labels_outside_the_known_classes(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    _write_bundle(data_dir, "imgs/first.png,not-a-class\n", "imgs/second.png\n")

    with pytest.raises(ValueError, match="label"):
        distribute(str(data_dir), str(output_dir))


@pytest.mark.parametrize("path_img", ["/etc/hostname", "../../secret.png", ""])
def test_distribute_rejects_train_path_img_that_escapes_the_data_directory(tmp_path, path_img):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    _write_bundle(data_dir, f"imgs/first.png,0\n{path_img},1\n", "imgs/second.png\n")

    with pytest.raises(ValueError, match="path_img"):
        distribute(str(data_dir), str(output_dir))

    assert _outside_files(output_dir, data_dir) == []


@pytest.mark.parametrize("path_img", ["/etc/passwd", "../../secret.png", "imgs/../../../secret.png"])
def test_distribute_rejects_test_path_img_that_escapes_the_data_directory(tmp_path, path_img):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    _write_bundle(data_dir, "imgs/first.png,0\n", f"imgs/second.png\n{path_img}\n")

    with pytest.raises(ValueError, match="path_img"):
        distribute(str(data_dir), str(output_dir))

    assert not (output_dir / "test" / "passwd").exists()
    assert _outside_files(output_dir, data_dir) == []


def test_distribute_rejects_symlinked_image_that_escapes_the_data_directory(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    _write_bundle(data_dir, "imgs/first.png,0\n", "imgs/second.png\n")
    shutil.move(str(data_dir / "imgs" / "first.png"), str(tmp_path / "moved.png"))
    (data_dir / "imgs" / "first.png").symlink_to(secret)

    with pytest.raises(ValueError, match="path_img"):
        distribute(str(data_dir), str(output_dir))

    assert not (output_dir / "test" / "secret.txt").exists()
