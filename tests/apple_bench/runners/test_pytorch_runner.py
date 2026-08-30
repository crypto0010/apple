"""Tests for PyTorchRunner."""

import numpy as np

from apple_bench.models import registry
from apple_bench.runners.pytorch_runner import PyTorchRunner


def test_pytorch_runner_returns_numpy_logits() -> None:
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    runner = PyTorchRunner(model, device="cpu")

    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    runner.warmup(x, n=2)
    out = runner.run(x)

    assert out.shape == (1, 4)
    assert out.dtype == np.float32
