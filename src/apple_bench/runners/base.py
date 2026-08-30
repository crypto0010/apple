"""Inference runner protocol shared across PyTorch, ORT, TensorRT."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Runner(Protocol):
    name: str

    def warmup(self, x: np.ndarray, n: int = 10) -> None: ...

    def run(self, x: np.ndarray) -> np.ndarray: ...
