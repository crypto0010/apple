"""Tests for the training harness using a toy model."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apple_bench.train.harness import TrainConfig, train_model


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):  # noqa: D401
        return self.fc(x)


def _toy_loaders(n: int = 64) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 8, generator=g)
    y = torch.randint(0, 4, (n,), generator=g)
    train_ds = TensorDataset(x[: n // 2], y[: n // 2])
    val_ds = TensorDataset(x[n // 2 :], y[n // 2 :])
    return DataLoader(train_ds, batch_size=8), DataLoader(val_ds, batch_size=8)


def test_harness_runs_and_writes_metrics(tmp_path: Path):
    train_loader, val_loader = _toy_loaders()
    cfg = TrainConfig(
        epochs=2, lr=1e-2, weight_decay=0.0, output_dir=tmp_path, device="cpu", amp=False,
    )
    result = train_model(_TinyModel(), train_loader, val_loader, cfg)

    assert result.best_val_acc is not None
    assert (tmp_path / "metrics.json").is_file()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics["epochs"]) == 2
    assert (tmp_path / "best.pt").is_file()


def test_harness_is_deterministic(tmp_path: Path):
    train_loader, val_loader = _toy_loaders()
    cfg = TrainConfig(
        epochs=1, lr=1e-2, weight_decay=0.0, output_dir=tmp_path / "a", device="cpu", amp=False,
    )
    r1 = train_model(_TinyModel(), train_loader, val_loader, cfg)
    cfg2 = TrainConfig(
        epochs=1, lr=1e-2, weight_decay=0.0, output_dir=tmp_path / "b", device="cpu", amp=False,
    )
    r2 = train_model(_TinyModel(), train_loader, val_loader, cfg2)
    assert abs(r1.best_val_acc - r2.best_val_acc) < 1e-6
