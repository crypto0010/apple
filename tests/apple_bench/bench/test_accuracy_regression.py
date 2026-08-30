"""Tests for FP32-vs-quantized accuracy regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from apple_bench.bench.accuracy_regression import regression_check


class _FixedRunner:
    name = "fixed"

    def __init__(self, fixed_pred: int, num_classes: int) -> None:
        self.fixed_pred = fixed_pred
        self.num_classes = num_classes

    def warmup(self, x: np.ndarray, n: int = 10) -> None:
        pass

    def run(self, x: np.ndarray) -> np.ndarray:
        bs = x.shape[0]
        logits = np.full((bs, self.num_classes), -10.0, dtype=np.float32)
        logits[:, self.fixed_pred] = 10.0
        return logits


def _make_loader(labels: list[int], num_classes: int) -> DataLoader:
    x = torch.zeros(len(labels), 3, 4, 4)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_regression_passes_when_quantized_matches_fp32() -> None:
    loader = _make_loader([0, 0, 0, 0], num_classes=2)
    fp32 = _FixedRunner(fixed_pred=0, num_classes=2)
    quant = _FixedRunner(fixed_pred=0, num_classes=2)
    report = regression_check(fp32, quant, loader, tolerance_pp=2.0)

    assert report.fp32_acc == 1.0
    assert report.quantized_acc == 1.0
    assert report.delta_pp == 0.0
    assert report.passes is True


def test_regression_fails_when_quantized_degrades(tmp_path: Path) -> None:
    loader = _make_loader([0, 0, 0, 0], num_classes=2)
    fp32 = _FixedRunner(fixed_pred=0, num_classes=2)  # 100% acc
    quant = _FixedRunner(fixed_pred=1, num_classes=2)  # 0% acc
    out_path = tmp_path / "report.json"
    report = regression_check(fp32, quant, loader, tolerance_pp=2.0, output_path=out_path)

    assert report.fp32_acc == 1.0
    assert report.quantized_acc == 0.0
    assert report.delta_pp == 100.0
    assert report.passes is False
    assert out_path.is_file()
