"""Generic training harness shared by all PyTorch models in this benchmark."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from apple_bench.config import SEED


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    output_dir: Path
    device: str = "cuda"
    amp: bool = True
    seed: int = SEED


@dataclass
class RunResult:
    best_val_acc: float
    best_epoch: int
    metrics_path: Path
    weights_path: Path
    config: dict = field(default_factory=dict)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _check_val(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Compute val-set accuracy. Sets the module to eval mode via train(False)."""
    model.train(mode=False)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> RunResult:
    _seed_all(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    model = model.to(device)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    history: list[dict] = []
    best_acc, best_epoch = -1.0, -1
    weights_path = cfg.output_dir / "best.pt"

    for epoch in range(cfg.epochs):
        model.train(mode=True)
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running_loss += loss.item() * y.numel()

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = _check_val(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), weights_path)

    metrics_path = cfg.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"epochs": history, "best": {"acc": best_acc, "epoch": best_epoch}}))

    return RunResult(
        best_val_acc=best_acc,
        best_epoch=best_epoch,
        metrics_path=metrics_path,
        weights_path=weights_path,
        config=asdict(cfg),
    )
