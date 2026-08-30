"""PlantVillage apple subset adapter.

Source: https://data.mendeley.com/datasets/tywbtsjrjv/1
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset

APPLE_CLASSES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

Split = Literal["train", "val"]


class PlantVillageApple(Dataset):
    """Apple subset of PlantVillage. Stratified train/val split, seeded for reproducibility."""

    def __init__(
        self,
        root: Path,
        split: Split,
        transform: Callable | None,
        train_ratio: float = 0.8,
        seed: int = 1729,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        samples: list[tuple[Path, int]] = []
        rng = random.Random(seed)
        for cls_idx, cls_name in enumerate(APPLE_CLASSES):
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Missing class dir: {cls_dir}")
            paths = sorted(p for p in cls_dir.iterdir() if p.is_file())
            if not paths:
                raise FileNotFoundError(f"No image files in class dir: {cls_dir}")
            rng.shuffle(paths)
            cut = int(len(paths) * train_ratio)
            chosen = paths[:cut] if split == "train" else paths[cut:]
            for p in chosen:
                samples.append((p, cls_idx))

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
