"""Project-wide configuration: paths, seeds, hardware assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RUNS_ROOT = PROJECT_ROOT / "runs"

# Determinism. Set everywhere; cuDNN benchmark off for reproducibility.
SEED = 1729

# Training defaults shared across models unless a model overrides.
@dataclass(frozen=True)
class TrainDefaults:
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4  # Orin Nano has 6 CPU cores; >4 risks DataLoader contention.
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4

TRAIN_DEFAULTS = TrainDefaults()
