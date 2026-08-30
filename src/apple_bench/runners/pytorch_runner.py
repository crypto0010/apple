"""PyTorch baseline runner (CPU or CUDA)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PyTorchRunner:
    name = "pytorch"

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.device = device
        self.model = model.to(device)
        self.model.train(mode=False)

    @torch.no_grad()
    def warmup(self, x: np.ndarray, n: int = 10) -> None:
        t = torch.from_numpy(x).to(self.device)
        for _ in range(n):
            self.model(t)
        if self.device == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def run(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x).to(self.device)
        out = self.model(t)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return out.cpu().numpy()
