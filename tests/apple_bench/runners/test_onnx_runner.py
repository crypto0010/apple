"""Tests for ONNXRunner — uses the to_onnx exporter to produce a model on the fly."""

from pathlib import Path

import numpy as np
import torch

from apple_bench.export.to_onnx import export_to_onnx
from apple_bench.models import registry
from apple_bench.runners.onnx_runner import ONNXRunner


def test_onnx_runner_returns_numpy_logits(tmp_path: Path) -> None:
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    onnx_path = tmp_path / "m.onnx"
    export_to_onnx(model, onnx_path, torch.randn(1, 3, 224, 224))

    runner = ONNXRunner(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    runner.warmup(x, n=2)
    out = runner.run(x)

    assert out.shape == (1, 4)
    assert out.dtype == np.float32
