"""Plant Pathology 2021 (FGVC8) field-captured apple leaf adapter.

Source: https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/data

Filters to single-label images whose label has a counterpart in the
PlantVillage apple class scheme. PlantVillage's Black_rot has no
mapped Plant Pathology counterpart and is intentionally absent
from this dataset; that asymmetry is reported as a known gap
in the reproduction report.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from apple_bench.data.plantvillage import APPLE_CLASSES

LABEL_TO_PV_CLASS: dict[str, str] = {
    "healthy": "Apple___healthy",
    "scab": "Apple___Apple_scab",
    "rust": "Apple___Cedar_apple_rust",
}

_PV_CLASS_TO_INDEX = {name: i for i, name in enumerate(APPLE_CLASSES)}


class PlantPathology2021(Dataset):
    """Field-captured apple imagery, mapped to PV's 4-class scheme."""

    def __init__(self, root: Path, transform: Callable | None) -> None:
        self.root = Path(root)
        self.transform = transform

        csv_path = self.root / "train.csv"
        images_dir = self.root / "train_images"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing labels CSV: {csv_path}")

        samples: list[tuple[Path, int]] = []
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            required_columns = {"image", "labels"}
            if not required_columns.issubset(reader.fieldnames or set()):
                missing = required_columns - set(reader.fieldnames or set())
                raise ValueError(f"CSV missing required columns: {sorted(missing)}")
            for row in reader:
                labels = row["labels"].split()
                if len(labels) != 1:
                    continue  # drop multi-label
                pv_class = LABEL_TO_PV_CLASS.get(labels[0])
                if pv_class is None:
                    continue
                img_path = images_dir / row["image"]
                if not img_path.is_file():
                    continue
                samples.append((img_path, _PV_CLASS_TO_INDEX[pv_class]))

        self.samples = samples
        self.targets = [t for _, t in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[object, int]:
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target
