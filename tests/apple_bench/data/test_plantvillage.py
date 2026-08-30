"""Tests for the PlantVillage adapter."""

from pathlib import Path

import pytest

from apple_bench.data.plantvillage import APPLE_CLASSES, PlantVillageApple


def test_apple_classes_exact():
    assert APPLE_CLASSES == [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
    ]


def test_dataset_indexes_only_apple_classes(tmp_path: Path):
    # Fixture: a synthetic PlantVillage layout with 2 apple imgs/class
    # and 1 unrelated class that must be ignored.
    root = tmp_path / "pv"
    for cls in APPLE_CLASSES:
        (root / cls).mkdir(parents=True)
        for i in range(2):
            (root / cls / f"{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG SOI/EOI
    (root / "Cherry___healthy").mkdir()
    (root / "Cherry___healthy" / "0.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    ds = PlantVillageApple(root=root, split="train", transform=None, train_ratio=1.0)
    assert len(ds) == 8
    assert set(ds.targets) == {0, 1, 2, 3}


def test_train_val_split_is_deterministic_and_disjoint(tmp_path: Path):
    root = tmp_path / "pv"
    for cls in APPLE_CLASSES:
        (root / cls).mkdir(parents=True)
        for i in range(20):
            (root / cls / f"{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    train1 = PlantVillageApple(root=root, split="train", transform=None, train_ratio=0.8)
    val1 = PlantVillageApple(root=root, split="val", transform=None, train_ratio=0.8)
    train2 = PlantVillageApple(root=root, split="train", transform=None, train_ratio=0.8)
    assert train1.samples == train2.samples  # determinism
    assert set(train1.samples).isdisjoint(set(val1.samples))  # disjoint
    assert len(train1) + len(val1) == 80


def test_non_files_in_class_dir_are_skipped(tmp_path: Path):
    """Accidental subdirectories should not be indexed."""
    root = tmp_path / "pv"
    for cls in APPLE_CLASSES:
        (root / cls).mkdir(parents=True)
        (root / cls / "0.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (root / cls / "1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (root / cls / "subdir").mkdir()                  # accidental subdirectory
        (root / cls / "subdir" / "x.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    ds = PlantVillageApple(root=root, split="train", transform=None, train_ratio=1.0)
    # Without is_file() filter, iterdir() would include the subdir, causing Image.open() to fail.
    # With is_file() filter: only 2 actual images per class.
    assert len(ds) == 2 * len(APPLE_CLASSES)


def test_empty_class_dir_raises(tmp_path: Path):
    root = tmp_path / "pv"
    for cls in APPLE_CLASSES:
        (root / cls).mkdir(parents=True)
        if cls != "Apple___healthy":
            (root / cls / "0.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    with pytest.raises(FileNotFoundError, match="No image files"):
        PlantVillageApple(root=root, split="train", transform=None, train_ratio=1.0)
