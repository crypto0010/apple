"""Tests for the Plant Pathology 2021 adapter."""

import csv
from pathlib import Path

import pytest

from apple_bench.data.plant_pathology import (
    LABEL_TO_PV_CLASS,
    PlantPathology2021,
)


def test_label_mapping_to_pv_classes():
    assert LABEL_TO_PV_CLASS == {
        "healthy": "Apple___healthy",
        "scab": "Apple___Apple_scab",
        "rust": "Apple___Cedar_apple_rust",
    }


def _make_fake_csv(root: Path) -> Path:
    csv_path = root / "train.csv"
    rows = [
        ("a.jpg", "healthy"),
        ("b.jpg", "scab"),
        ("c.jpg", "rust"),
        ("d.jpg", "powdery_mildew"),       # should be dropped (unmapped)
        ("e.jpg", "scab frog_eye_leaf_spot"),  # multi-label; dropped
    ]
    images_dir = root / "train_images"
    images_dir.mkdir(parents=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "labels"])
        for name, label in rows:
            w.writerow([name, label])
            (images_dir / name).write_bytes(b"\xff\xd8\xff\xd9")
    return csv_path


def test_filters_to_mapped_single_label(tmp_path: Path):
    _make_fake_csv(tmp_path)
    ds = PlantPathology2021(root=tmp_path, transform=None)
    assert len(ds) == 3
    targets = sorted(ds.targets)
    # PV class indices: 0=scab, 1=blackrot, 2=cedarrust, 3=healthy
    assert targets == [0, 2, 3]


def test_csv_missing_required_columns_raises(tmp_path: Path):
    """Adapter must reject CSVs that lack 'image' or 'labels' columns with a clear error."""
    images_dir = tmp_path / "train_images"
    images_dir.mkdir()
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "category"])      # wrong column names
        w.writerow(["a.jpg", "scab"])
    with pytest.raises(ValueError, match="missing required columns"):
        PlantPathology2021(root=tmp_path, transform=None)
