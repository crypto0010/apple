"""AppleLeaf9 cross-domain adapter (mapped subset).

Source: https://github.com/JasonYangCode/AppleLeaf9

Restricted to the 3 classes that map cleanly into PV's 4-class apple scheme.
The other 6 AppleLeaf9 classes (alternaria, mosaic, brown_spot, grey_spot,
anthracnose, plus the PV-derived 'general healthy') are intentionally dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from apple_bench.data.plantvillage import APPLE_CLASSES

APPLELEAF9_TO_PV: dict[str, str] = {
    "Health": "Apple___healthy",
    "Scab": "Apple___Apple_scab",
    "Rust": "Apple___Cedar_apple_rust",
}

_PV_CLASS_TO_INDEX = {name: i for i, name in enumerate(APPLE_CLASSES)}


class AppleLeaf9(Dataset):
    """AppleLeaf9 restricted to classes that map into the PV 4-class scheme."""

    def __init__(self, root: Path, transform: Callable | None) -> None:
        self.root = Path(root)
        self.transform = transform

        samples: list[tuple[Path, int]] = []
        any_class_found = False
        for src_class, pv_class in APPLELEAF9_TO_PV.items():
            cls_dir = self.root / src_class
            if not cls_dir.is_dir():
                continue
            any_class_found = True
            paths = sorted(p for p in cls_dir.iterdir() if p.is_file())
            for path in paths:
                samples.append((path, _PV_CLASS_TO_INDEX[pv_class]))

        if not any_class_found:
            raise FileNotFoundError(
                f"No mapped AppleLeaf9 class dirs found under {self.root}. "
                f"Expected one of: {sorted(APPLELEAF9_TO_PV.keys())}"
            )

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
