"""Compare two runners' accuracy on the same eval loader.

Used to verify that an INT8-quantized runner has not regressed
meaningfully versus its FP32 sibling. ``regression_check`` returns a
percentage-point delta and a pass/fail flag against a configurable
tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from apple_bench.runners.base import Runner


@dataclass
class RegressionReport:
    fp32_acc: float
    quantized_acc: float
    delta_pp: float
    passes: bool


def regression_check(
    fp32_runner: Runner,
    quantized_runner: Runner,
    loader: DataLoader,
    tolerance_pp: float = 2.0,
    output_path: Path | None = None,
) -> RegressionReport:
    fp32_acc = _eval_runner(fp32_runner, loader)
    q_acc = _eval_runner(quantized_runner, loader)
    delta = (fp32_acc - q_acc) * 100  # percentage points
    report = RegressionReport(
        fp32_acc=fp32_acc,
        quantized_acc=q_acc,
        delta_pp=delta,
        passes=delta <= tolerance_pp,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.__dict__, indent=2))
    return report


def _eval_runner(runner: Runner, loader: DataLoader) -> float:
    correct, total = 0, 0
    for x, y in loader:
        logits = runner.run(x.numpy().astype(np.float32))
        pred = logits.argmax(axis=1)
        correct += int((pred == y.numpy()).sum())
        total += int(y.numel())
    return correct / max(total, 1)
