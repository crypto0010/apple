"""Cross-domain scoring with bootstrapped accuracy CI and confusion matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def score_with_ci(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str,
    output_path: Path,
    n_bootstrap: int = 1000,
    seed: int = 1729,
) -> dict:
    """Run model over loader (no grad), compute accuracy + bootstrapped CI."""
    model.train(mode=False)
    model.to(device)
    all_pred: list[int] = []
    all_true: list[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())
        all_true.extend(y.tolist())

    pred = np.asarray(all_pred)
    true = np.asarray(all_true)
    correct = (pred == true).astype(np.int32)
    accuracy = float(correct.mean())

    # Bootstrap 95% CI.
    rng = np.random.default_rng(seed)
    n = correct.shape[0]
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = correct[idx].mean()
    ci_lower, ci_upper = np.quantile(boots, [0.025, 0.975]).tolist()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred, strict=True):
        cm[t, p] += 1

    report = {
        "n_samples": int(n),
        "accuracy": accuracy,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confusion_matrix": cm.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report
